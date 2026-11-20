import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class ModelManager:
    """Manages loading, device placement, and inference for Waste Detection and Pose Estimation models."""

    def __init__(
        self,
        weights_path: str = "best.pt",
        hf_repo_id: str = "",
        hf_filename: str = "best.pt",
        pose_model_name: str = "yolo11n-pose.pt",
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)
        print(f"[INFO] Using compute device: {self.device}")

        # Optimize CPU threads to prevent thread contention & reduce latency
        if self.device == "cpu":
            num_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            print(f"[INFO] CPU inference threads set to: {num_threads}")

        # 1. Load Custom Local Waste Model (Priority 1) or Hugging Face Fallback (Priority 2)
        local_target = Path(weights_path if weights_path else hf_filename)
        if local_target.exists():
            print(f"[INFO] Loading custom local waste model: {local_target.resolve()} ...")
            self.waste_model = YOLO(str(local_target))
            self.waste_model.to(self.device)
            print(f"[INFO] Custom waste model active! Classes ({len(self.waste_model.names)}): {list(self.waste_model.names.values())}")
        elif hf_repo_id:
            print(f"[INFO] Local model not found. Downloading from Hugging Face ({hf_repo_id})...")
            try:
                downloaded_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
                self.waste_model = YOLO(downloaded_path)
                self.waste_model.to(self.device)
                print(f"[INFO] Waste model loaded from HF: {downloaded_path}")
                print(f"[INFO] Waste classes ({len(self.waste_model.names)}): {list(self.waste_model.names.values())}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model from Hugging Face: {e}")
        else:
            raise RuntimeError(
                f"Model weights file '{local_target}' not found. Please ensure 'best.pt' is in the project root."
            )

        # 2. Load Pose Estimation model (COCO 17 keypoints)
        print(f"[INFO] Loading pose estimation model ({pose_model_name})...")
        try:
            self.pose_model = YOLO(pose_model_name).to(self.device)
        except Exception:
            print("[INFO] Falling back to yolov8n-pose.pt...")
            self.pose_model = YOLO("yolov8n-pose.pt").to(self.device)

        print("[INFO] Model bootstrap completed successfully.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def predict_pose(self, frame, conf: float = 0.40, kpt_conf: float = 0.25, imgsz: int = 640) -> List[Dict[str, Any]]:
        """
        Runs pose estimation with torch.inference_mode.
        Returns detected persons with bounding box, score, and wrist keypoints.
        COCO keypoints: 9 = left wrist, 10 = right wrist.
        """
        with torch.inference_mode():
            results = self.pose_model(frame, verbose=False, conf=conf, imgsz=imgsz, device=self.device)[0]
        persons = []

        if results.boxes is None or len(results.boxes) == 0:
            return persons

        has_kpts = results.keypoints is not None and results.keypoints.data is not None

        for i, box_data in enumerate(results.boxes):
            box = list(map(int, box_data.xyxy[0].cpu().numpy()))
            score = float(box_data.conf[0].cpu().numpy())

            wrists = []
            all_kpts = []
            if has_kpts and i < len(results.keypoints.data):
                kpts = results.keypoints.data[i].cpu().numpy()
                all_kpts = kpts
                # Keypoint 9 = left wrist, 10 = right wrist
                for wrist_idx in (9, 10):
                    if wrist_idx < len(kpts):
                        kx, ky, kconf = kpts[wrist_idx]
                        if kconf >= kpt_conf:
                            wrists.append((int(kx), int(ky)))

            persons.append({
                "box": box,
                "conf": score,
                "wrists": wrists,
                "keypoints": all_kpts,
            })

        return persons

    def predict_waste(self, frame, conf: float = 0.15, imgsz: int = 640) -> List[Dict[str, Any]]:
        """
        Runs waste detection with torch.inference_mode.
        Returns detected litter objects with bounding box, class label, and score.
        """
        with torch.inference_mode():
            results = self.waste_model(frame, verbose=False, conf=conf, imgsz=imgsz, device=self.device)[0]
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for b in results.boxes:
            box = list(map(int, b.xyxy[0].cpu().numpy()))
            cls_id = int(b.cls[0].cpu().numpy())
            label = self.waste_model.names.get(cls_id, f"waste_{cls_id}")
            score = float(b.conf[0].cpu().numpy())

            detections.append({
                "box": box,
                "label": label,
                "conf": score,
            })

        return detections
