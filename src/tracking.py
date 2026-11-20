import math
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.association import box_center, box_iou


class TrackState(Enum):
    HELD = "HELD"                    # Actively carried or held near hand
    UNATTENDED = "UNATTENDED"        # Dropped/separated, countdown in progress
    CONFIRMED = "CONFIRMED"          # Confirmed littering event
    GROUND_EXISTING = "PREEXISTING"  # Ground trash present prior to observation (ignore)


class TrackedItem:
    def __init__(
        self,
        track_id: int,
        box: List[int],
        label: str,
        is_held: bool,
        hand_pt: Optional[Tuple[int, int]] = None,
    ):
        self.track_id = track_id
        self.box = box
        self.label = label
        self.lost = 0
        self.hand_pt = hand_pt

        # Possession history
        self.held_frames = 1 if is_held else 0
        self.unattended_frames = 0
        self.was_possessed = False   # Must be held for >= min_held_frames to qualify
        self.state = TrackState.HELD if is_held else TrackState.GROUND_EXISTING
        self.alert_triggered = False

        # Trajectory history for motion & downward drop analysis
        self.centroids = deque(maxlen=30)
        self.centroids.append(box_center(box))

    def update_held(self, box: List[int], hand_pt: Optional[Tuple[int, int]], min_held_frames: int):
        self.box = box
        self.lost = 0
        self.hand_pt = hand_pt
        self.held_frames += 1
        self.unattended_frames = 0
        self.centroids.append(box_center(box))

        if self.held_frames >= min_held_frames:
            self.was_possessed = True
            self.state = TrackState.HELD

    def update_unattended(self, box: List[int], alert_threshold: int):
        self.box = box
        self.lost = 0
        self.hand_pt = None
        self.centroids.append(box_center(box))

        # Only transition to unattended countdown if this item was verified to be
        # possessed/carried by a person. Prevents false alerts on preexisting litter!
        if self.was_possessed:
            self.unattended_frames += 1
            if self.unattended_frames >= alert_threshold:
                self.state = TrackState.CONFIRMED
                self.alert_triggered = True
            else:
                self.state = TrackState.UNATTENDED
        else:
            self.state = TrackState.GROUND_EXISTING


class LitterTracker:
    """
    Robust Multi-Object State Tracker for Littering Detection.
    Solves the 'Innocent Bystander' problem by verifying prior hand possession.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        unattended_frames_alert: int = 30,
        track_lost_ttl: int = 20,
        min_held_frames: int = 4,
    ):
        self.iou_threshold = iou_threshold
        self.unattended_frames_alert = unattended_frames_alert
        self.track_lost_ttl = track_lost_ttl
        self.min_held_frames = min_held_frames

        self._next_id = 1
        self.tracks: Dict[int, TrackedItem] = {}

    def update(self, detections: List[Tuple[List[int], str, bool, Optional[Tuple[int, int]]]]):
        """
        detections: list of tuples: (box, label, is_held, hand_pt)
        """
        # Age all tracks
        for track in self.tracks.values():
            track.lost += 1

        matched_tracks = set()
        matched_dets = set()

        # Step 1: Match existing tracks using IoU & centroid proximity
        for det_idx, (box, label, is_held, hand_pt) in enumerate(detections):
            best_id = None
            best_score = self.iou_threshold

            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue

                iou = box_iou(box, track.box)
                if iou > best_score:
                    best_score = iou
                    best_id = tid

            # Fallback distance matching for small fast-moving objects
            if best_id is None:
                cx, cy = box_center(box)
                min_dist = 60.0 # max centroid jump in pixels
                for tid, track in self.tracks.items():
                    if tid in matched_tracks:
                        continue
                    tcx, tcy = track.centroids[-1]
                    dist = math.hypot(cx - tcx, cy - tcy)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = tid

            if best_id is not None:
                track = self.tracks[best_id]
                if is_held:
                    track.update_held(box, hand_pt, self.min_held_frames)
                else:
                    track.update_unattended(box, self.unattended_frames_alert)

                matched_tracks.add(best_id)
                matched_dets.add(det_idx)

        # Step 2: Register new detections
        for det_idx, (box, label, is_held, hand_pt) in enumerate(detections):
            if det_idx in matched_dets:
                continue

            # Only initialize an active tracked item if it is actively held or in hand interaction.
            # If an item appears out of nowhere on the ground, initialize as GROUND_EXISTING.
            new_track = TrackedItem(
                track_id=self._next_id,
                box=box,
                label=label,
                is_held=is_held,
                hand_pt=hand_pt,
            )
            self.tracks[self._next_id] = new_track
            self._next_id += 1

        # Step 3: Evict stale tracks
        stale_ids = [tid for tid, t in self.tracks.items() if t.lost > self.track_lost_ttl]
        for tid in stale_ids:
            del self.tracks[tid]

        return self.tracks
