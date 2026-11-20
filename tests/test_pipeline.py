"""Smoke and unit tests for Litter Watch AI components."""

import numpy as np
import pytest

from src.association import box_center, box_iou, edge_dist, evaluate_hand_association
from src.recorder import IncidentRecorder
from src.tracking import LitterTracker, TrackState


def test_box_iou():
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert pytest.approx(box_iou(box1, box2)) == 1.0

    box3 = [60, 60, 100, 100]
    assert box_iou(box1, box3) == 0.0


def test_edge_dist():
    box1 = [0, 0, 10, 10]
    box2 = [20, 0, 30, 10]
    assert edge_dist(box1, box2) == 10.0


def test_wrist_hand_association():
    # Simulated person with detected wrists
    person = {
        "box": [100, 50, 200, 350],  # height = 300
        "conf": 0.85,
        "wrists": [(150, 200)],       # hand position
    }
    
    # Waste object held right near the wrist
    waste_box_held = [145, 195, 160, 210]
    is_held, p, hand_pt = evaluate_hand_association(
        waste_box=waste_box_held,
        persons=[person],
        hand_distance_ratio=0.22,
    )
    assert is_held is True
    assert hand_pt == (150, 200)

    # Waste object far away on the floor
    waste_box_far = [400, 400, 420, 420]
    is_held_far, _, _ = evaluate_hand_association(
        waste_box=waste_box_far,
        persons=[person],
        hand_distance_ratio=0.22,
    )
    assert is_held_far is False


def test_innocent_bystander_prevention():
    """
    Verifies that a pre-existing object on the ground DOES NOT trigger a littering alert
    when a pedestrian simply walks past it without possessing it.
    """
    tracker = LitterTracker(
        iou_threshold=0.20,
        unattended_frames_alert=5,
        min_held_frames=4,
    )

    ground_can = [100, 300, 120, 320]

    # Frame 1: Can appears on ground (is_held = False)
    tracks = tracker.update([(ground_can, "can", False, None)])
    tid = list(tracks.keys())[0]
    assert tracks[tid].state == TrackState.GROUND_EXISTING

    # Frames 2-10: Can remains unattended on ground
    for _ in range(10):
        tracks = tracker.update([(ground_can, "can", False, None)])

    # MUST NOT trigger confirmed litter alert because it was never held!
    assert tracks[tid].state == TrackState.GROUND_EXISTING
    assert tracks[tid].alert_triggered is False


def test_held_and_dropped_workflow():
    """
    Verifies that an object actively held and then left unattended triggers CONFIRMED litter alert.
    """
    tracker = LitterTracker(
        iou_threshold=0.20,
        unattended_frames_alert=4,
        min_held_frames=3,
    )

    box = [100, 100, 120, 120]

    # Held for 3 frames -> establishes verified possession
    for _ in range(3):
        tracks = tracker.update([(box, "cup", True, (110, 110))])

    tid = list(tracks.keys())[0]
    assert tracks[tid].state == TrackState.HELD
    assert tracks[tid].was_possessed is True

    # Now dropped / separated
    for i in range(3):
        tracks = tracker.update([(box, "cup", False, None)])
        assert tracks[tid].state == TrackState.UNATTENDED

    # 4th unattended frame reaches alert threshold (4 frames)
    tracks = tracker.update([(box, "cup", False, None)])
    assert tracks[tid].state == TrackState.CONFIRMED
    assert tracks[tid].alert_triggered is True


def test_recorder_buffer(tmp_path):
    recorder = IncidentRecorder(output_dir=str(tmp_path), buffer_seconds=1, fps=10.0)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(15):
        recorder.push_frame(fake_frame)

    assert len(recorder.buffer) == 10  # Maxlen reached

    incident_id = recorder.trigger_incident(
        track_id=1,
        label="bottle",
        box=[10, 10, 30, 30],
        annotated_frame=fake_frame,
    )
    assert incident_id is not None
    assert recorder.total_incidents == 1
