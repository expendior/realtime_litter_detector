import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2


class IncidentRecorder:
    """
    Automated Incident Evidence Vault.
    Maintains a circular FIFO buffer of surveillance frames.
    When a littering violation is confirmed, asynchronously packages:
      1. A 5-second video clip (.mp4) capturing the drop action
      2. A high-resolution snapshot (.jpg)
      3. A structured JSON audit record (.json)
    """

    def __init__(self, output_dir: str = "incidents", buffer_seconds: int = 5, fps: float = 30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.buffer_len = int(buffer_seconds * fps)
        self.buffer = deque(maxlen=self.buffer_len)
        self.recorded_track_ids: Set[int] = set()
        self.incident_log: List[Dict[str, Any]] = []

    def push_frame(self, frame):
        """Append latest frame into circular buffer."""
        self.buffer.append(frame.copy())

    def trigger_incident(
        self,
        track_id: int,
        label: str,
        box: List[int],
        annotated_frame=None,
    ) -> Optional[str]:
        """
        Triggers an incident package export if not already recorded.
        Runs video encoding in a daemon thread so it doesn't drop surveillance frames!
        """
        if track_id in self.recorded_track_ids:
            return None

        self.recorded_track_ids.add(track_id)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        incident_id = f"INC-{track_id:03d}-{timestamp_str}"

        clip_filename = f"{incident_id}.mp4"
        snap_filename = f"{incident_id}.jpg"
        meta_filename = f"{incident_id}.json"

        clip_path = self.output_dir / clip_filename
        snap_path = self.output_dir / snap_filename
        meta_path = self.output_dir / meta_filename

        # Snapshot saving
        if annotated_frame is not None:
            cv2.imwrite(str(snap_path), annotated_frame)
        elif self.buffer:
            cv2.imwrite(str(snap_path), self.buffer[-1])

        # Buffer snapshot for thread
        frames_to_write = list(self.buffer)

        # Asynchronous clip export
        def _write_clip():
            if not frames_to_write:
                return
            h, w = frames_to_write[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))
            for f in frames_to_write:
                writer.write(f)
            writer.release()
            print(f"[EVIDENCE] Logged incident clip -> {clip_path}")

        writer_thread = threading.Thread(target=_write_clip, daemon=True)
        writer_thread.start()

        # Metadata export
        metadata = {
            "incident_id": incident_id,
            "track_id": track_id,
            "label": label,
            "bounding_box": box,
            "timestamp": datetime.now().isoformat(),
            "video_evidence": str(clip_path),
            "snapshot_evidence": str(snap_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.incident_log.append(metadata)
        return incident_id

    @property
    def total_incidents(self) -> int:
        return len(self.recorded_track_ids)
