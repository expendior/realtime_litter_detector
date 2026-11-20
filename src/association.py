import math
from typing import Any, Dict, List, Optional, Tuple


def box_iou(a: List[int], b: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_overlap(b1: List[int], b2: List[int]) -> bool:
    """Checks if two bounding boxes intersect/overlap."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def box_center(box: List[int]) -> Tuple[int, int]:
    """Returns the (cx, cy) center coordinate of a bounding box."""
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def edge_dist(b1: List[int], b2: List[int]) -> float:
    """Computes minimum Euclidean edge-to-edge distance between two boxes."""
    dx = max(b2[0] - b1[2], 0, b1[0] - b2[2])
    dy = max(b2[1] - b1[3], 0, b1[0] - b2[3])
    return math.hypot(dx, dy)


def is_on_surface(box: List[int], frame_height: int, ground_ratio: float = 0.15) -> bool:
    """
    Checks if object bottom is low enough in the frame to plausibly be on a surface/ground
    rather than floating in the sky.
    """
    return box[3] > frame_height * ground_ratio


def evaluate_hand_association(
    waste_box: List[int],
    persons: List[Dict[str, Any]],
    hand_distance_ratio: float = 0.38,
    fallback_dist_px: float = 90.0,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Tuple[int, int]]]:
    """
    Determines whether a waste object is actively held or in close possession of a person.
    
    A waste object is classified as HELD / POSSESSED if:
      1. It intersects or overlaps the person's upper/mid body (lap, chest, hands).
      2. Its center is within hand_distance_ratio of a detected wrist or hand keypoint.
      3. It is within immediate arm's reach of the person's body (above the feet).
      
    An object is ONLY unattended if it has physically separated from the person's body and reach!
    """
    if not persons:
        return False, None, None

    wcx, wcy = box_center(waste_box)
    best_person = None
    best_hand_pt = None
    min_dist = float("inf")

    for p in persons:
        p_box = p["box"]
        p_height = max(p_box[3] - p_box[1], 1)
        wrists = p.get("wrists", [])
        feet_threshold = p_box[3] - (p_height * 0.12)  # Bottom 12% is feet line

        # 1. Direct Overlap: If waste box touches or overlaps the person above the feet line
        if box_overlap(waste_box, p_box) and wcy < feet_threshold:
            best_person = p
            if wrists:
                best_hand_pt = min(wrists, key=lambda pt: math.hypot(wcx - pt[0], wcy - pt[1]))
            else:
                best_hand_pt = (wcx, wcy)
            return True, best_person, best_hand_pt

        # 2. Wrist / Hand Keypoint Proximity Check (generous reach for fingers & outstretched arms)
        if wrists:
            for (hx, hy) in wrists:
                dist = math.hypot(wcx - hx, wcy - hy)
                norm_dist = dist / p_height
                if norm_dist < hand_distance_ratio and norm_dist < min_dist:
                    min_dist = norm_dist
                    best_person = p
                    best_hand_pt = (hx, hy)

        # 3. Arm Reach & Edge Distance Check (covers outstretched fingers or occluded wrists)
        edist = edge_dist(waste_box, p_box)
        norm_edist = edist / p_height
        reach_threshold = max(hand_distance_ratio, 0.35)
        if norm_edist < reach_threshold and wcy < feet_threshold:
            if norm_edist < min_dist:
                min_dist = norm_edist
                best_person = p
                if wrists:
                    best_hand_pt = min(wrists, key=lambda pt: math.hypot(wcx - pt[0], wcy - pt[1]))
                else:
                    best_hand_pt = (wcx, wcy)

    is_held = best_person is not None
    return is_held, best_person, best_hand_pt
