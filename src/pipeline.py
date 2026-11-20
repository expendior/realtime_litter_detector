import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.association import evaluate_hand_association, is_on_surface
from src.models import ModelManager
from src.recorder import IncidentRecorder
from src.tracking import LitterTracker, TrackState
from src.ui import SurveillanceRenderer


class ThreadedCamera:
    """
    Non-blocking background video capture thread.
    Always provides the freshest camera frame to eliminate 1-2 second buffer lag.
    """

    def __init__(self, source: Union[int, str], width: int = 1280, height: int = 720, use_dshow: bool = True):
        self.is_camera = isinstance(source, int)
        if self.is_camera and use_dshow:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source!r}")

        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.stopped = False
        self.ret, self.frame = self.cap.read()

        if self.is_camera:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.ret = ret
            self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_camera:
            return self.ret, (self.frame.copy() if self.frame is not None else None)
        return self.cap.read()

    def release(self):
        self.stopped = True
        self.cap.release()

    def get(self, prop: int) -> float:
        return self.cap.get(prop)


class LitterDetectionPipeline:
    """End-to-end Computer Vision Pipeline for Real-Time Littering Detection & Evidence Logging."""

    def __init__(
        self,
        config: Dict[str, Any],
        waste_conf_override: float = None,
        enable_record: bool = True,
        headless: bool = False,
        output_path: Optional[str] = None,
        skip_override: Optional[int] = None,
        imgsz_override: Optional[int] = None,
    ):
        self.config = config
        self.headless = headless
        self.output_path = output_path

        # Override threshold if passed from CLI
        if waste_conf_override is not None:
            self.config["thresholds"]["waste_conf"] = waste_conf_override

        # 1. Performance Settings
        perf_cfg = self.config.get("performance", {})
        self.process_every_n_frames = (
            skip_override if skip_override is not None else perf_cfg.get("process_every_n_frames", 2)
        )
        self.inference_imgsz = (
            imgsz_override if imgsz_override is not None else perf_cfg.get("inference_imgsz", 640)
        )
        print(f"[INFO] Performance: Processing neural nets every {self.process_every_n_frames} frames @ {self.inference_imgsz}px")

        # 2. Models
        model_cfg = self.config.get("models", {})
        self.model_manager = ModelManager(
            weights_path=model_cfg.get("weights_path", "best.pt"),
            hf_repo_id=model_cfg.get("hf_repo_id", ""),
            hf_filename=model_cfg.get("hf_filename", "best.pt"),
            pose_model_name=model_cfg.get("pose_model", "yolo11n-pose.pt"),
            device=model_cfg.get("device", "auto"),
        )

        # 3. Association & Tracking parameters
        inter_cfg = self.config.get("interaction", {})
        thresh_cfg = self.config.get("thresholds", {})

        self.person_conf = thresh_cfg.get("person_conf", 0.35)
        self.waste_conf = thresh_cfg.get("waste_conf", 0.15)
        self.kpt_conf = thresh_cfg.get("keypoint_conf", 0.25)

        self.hand_distance_ratio = inter_cfg.get("hand_distance_ratio", 0.24)
        self.fallback_dist_px = inter_cfg.get("fallback_held_distance_px", 70.0)
        self.min_held_frames = inter_cfg.get("min_held_frames", 3)
        self.unattended_frames_alert = inter_cfg.get("unattended_frames_alert", 25)
        self.ground_ratio = inter_cfg.get("ground_ratio", 0.12)

        self.tracker = LitterTracker(
            iou_threshold=inter_cfg.get("iou_match_threshold", 0.20),
            unattended_frames_alert=self.unattended_frames_alert,
            track_lost_ttl=inter_cfg.get("track_lost_ttl", 20),
            min_held_frames=self.min_held_frames,
        )

        # 4. Evidence Recorder
        rec_cfg = self.config.get("recorder", {})
        self.enable_record = enable_record and rec_cfg.get("enabled", True)
        self.recorder = IncidentRecorder(
            output_dir=rec_cfg.get("output_dir", "incidents"),
            buffer_seconds=rec_cfg.get("buffer_seconds", 5),
            fps=rec_cfg.get("fps", 30.0),
        )

        # 5. Renderer
        self.renderer = SurveillanceRenderer(config)

    def run(self, source: Union[int, str] = 0):
        """Runs the surveillance detection loop on camera index or video file."""
        cam_cfg = self.config.get("camera", {})
        use_dshow = cam_cfg.get("use_dshow", True)
        width = cam_cfg.get("width", 1280)
        height = cam_cfg.get("height", 720)

        # Initialize threaded camera for real-time video without frame buffer lag
        cap = ThreadedCamera(source=source, width=width, height=height, use_dshow=use_dshow)

        frame_idx = 0
        t_prev = time.perf_counter()
        fps = 30.0

        # Cached detections for frame-skipping acceleration
        cached_persons: List[Dict[str, Any]] = []
        cached_waste: List[Dict[str, Any]] = []

        print(f"[INFO] Surveillance stream active on {source}. Press 'Q' or ESC to exit.")

        video_writer = None
        if self.output_path:
            out_p = Path(self.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            total_w = width + self.renderer.panel_width
            total_h = height
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(out_p), fourcc, 30.0, (total_w, total_h))
            print(f"[INFO] Exporting annotated surveillance feed to {self.output_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If camera, wait briefly; if video file, break on EOF
                    if isinstance(source, int):
                        time.sleep(0.01)
                        continue
                    print("[INFO] Video stream ended or EOF reached.")
                    break

                fh, fw = frame.shape[:2]
                frame_idx += 1

                # Step A: Push raw unannotated frame to evidence buffer
                if self.enable_record:
                    self.recorder.push_frame(frame)

                # Step B: Render subtle scanline effect
                self.renderer.draw_scanline(frame, frame_idx)

                # Step C & D: Multi-Rate Neural Inference
                # Run heavy deep models every N frames to keep FPS high & prevent CPU lag
                if frame_idx % self.process_every_n_frames == 0 or not cached_persons:
                    cached_persons = self.model_manager.predict_pose(
                        frame,
                        conf=self.person_conf,
                        kpt_conf=self.kpt_conf,
                        imgsz=self.inference_imgsz,
                    )
                    cached_waste = self.model_manager.predict_waste(
                        frame,
                        conf=self.waste_conf,
                        imgsz=self.inference_imgsz,
                    )

                persons = cached_persons
                waste_dets = cached_waste

                # Step E: Spatial Hand Association
                tracker_inputs = []
                for w in waste_dets:
                    box = w["box"]
                    label = w["label"]

                    is_held, person, hand_pt = evaluate_hand_association(
                        waste_box=box,
                        persons=persons,
                        hand_distance_ratio=self.hand_distance_ratio,
                        fallback_dist_px=self.fallback_dist_px,
                    )

                    if is_held:
                        tracker_inputs.append((box, label, True, hand_pt))
                    elif is_on_surface(box, fh, self.ground_ratio):
                        tracker_inputs.append((box, label, False, None))

                # Step F: Multi-Object State Tracking
                active_tracks = self.tracker.update(tracker_inputs)

                # Step G: Render Persons
                for p in persons:
                    self.renderer.draw_person(frame, p)

                # Step H: Render Active Tracks & Trigger Incident Capture
                n_confirmed_litter = 0
                for tid, track in active_tracks.items():
                    if track.lost > 0:
                        continue

                    is_confirmed = self.renderer.draw_track_item(
                        frame,
                        track,
                        frame_idx,
                        self.unattended_frames_alert,
                    )

                    if is_confirmed:
                        n_confirmed_litter += 1
                        if self.enable_record:
                            self.recorder.trigger_incident(
                                track_id=tid,
                                label=track.label,
                                box=track.box,
                                annotated_frame=frame,
                            )

                # Step I: Compute Rolling FPS
                t_now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now

                # Step J: Assemble Wide Surveillance Canvas (Feed + Sidebar)
                pw = self.renderer.panel_width
                canvas = np.zeros((fh, fw + pw, 3), dtype=np.uint8)
                canvas[0:fh, 0:fw] = frame

                self.renderer.draw_sidebar(
                    canvas,
                    fw=fw,
                    fh=fh,
                    fps=fps,
                    n_persons=len(persons),
                    n_litter=n_confirmed_litter,
                    total_incidents=self.recorder.total_incidents,
                    tracks=active_tracks,
                    frame_idx=frame_idx,
                )

                # Step K: Top Alert Banner & Watermark
                self.renderer.draw_alert_banner(canvas, n_confirmed_litter, frame_idx, fw + pw)
                self.renderer.draw_watermark(canvas, fh, fw)

                # Step L: Write to output file if requested
                if video_writer is not None:
                    # Ensure canvas size matches writer setup
                    if video_writer.isOpened():
                        video_writer.write(canvas)

                # Step M: Display UI window
                if not self.headless:
                    try:
                        cv2.imshow("LITTER WATCH AI  ·  Surveillance HUD", canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                            break
                    except cv2.error as e:
                        print(f"\n[WARNING] OpenCV Display error: {e}")
                        print("[INFO] Falling back to headless mode. To export annotated video, run with: --output result.mp4")
                        self.headless = True

        finally:
            if video_writer is not None:
                video_writer.release()
            cap.release()
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            print("[INFO] Pipeline shut down cleanly.")
            print(f"[SUMMARY] Total Incidents Captured: {self.recorder.total_incidents}")
