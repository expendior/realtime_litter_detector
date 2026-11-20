import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracking import TrackState, TrackedItem


class SurveillanceRenderer:
    """Renders cyber-themed surveillance HUD overlays, bounding boxes, and telemetry sidebar."""

    def __init__(self, config: Dict[str, Any]):
        ui_cfg = config.get("ui", {})
        self.accent_cyan = tuple(ui_cfg.get("accent_cyan", [220, 200, 0]))
        self.green_safe = tuple(ui_cfg.get("green_safe", [80, 220, 80]))
        self.amber_warning = tuple(ui_cfg.get("amber_warning", [30, 165, 255]))
        self.red_alert = tuple(ui_cfg.get("red_alert", [40, 40, 255]))
        self.red_bright = tuple(ui_cfg.get("red_bright", [80, 80, 255]))
        self.dark_bg = tuple(ui_cfg.get("dark_bg", [12, 12, 14]))
        self.panel_bg = tuple(ui_cfg.get("panel_bg", [18, 20, 22]))
        self.panel_line = tuple(ui_cfg.get("panel_line", [45, 48, 52]))
        self.panel_width = ui_cfg.get("panel_width", 250)
        self.white = (255, 255, 255)

    def alpha_rect(self, img, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float = 0.45):
        """Semi-transparent filled rectangle overlay."""
        x1 = max(0, min(x1, img.shape[1] - 1))
        x2 = max(0, min(x2, img.shape[1]))
        y1 = max(0, min(y1, img.shape[0] - 1))
        y2 = max(0, min(y2, img.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return

        sub = img[y1:y2, x1:x2]
        overlay = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

    def corner_box(self, img, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], thickness: int = 2, arm: int = 14):
        """Draws surveillance corner brackets on bounding box."""
        pts = [
            ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
            ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
            ((x1, y2 - arm), (x1, y2), (x1 + arm, y2)),
            ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),
        ]
        for p0, corner, p1 in pts:
            cv2.line(img, p0, corner, color, thickness, cv2.LINE_AA)
            cv2.line(img, corner, p1, color, thickness, cv2.LINE_AA)

    def label_chip(self, img, text: str, x: int, y: int, color: Tuple[int, int, int], bg_alpha: float = 0.70, scale: float = 0.42, pad: int = 4):
        """Pill-shaped label badge with semi-transparent background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        bx1 = max(0, x)
        by1 = max(0, y - th - pad * 2)
        bx2 = bx1 + tw + pad * 2
        by2 = max(by1 + th + pad * 2, y)

        self.alpha_rect(img, bx1, by1, bx2, by2, color, bg_alpha)
        cv2.putText(img, text, (bx1 + pad, by2 - pad - 1), font, scale, self.white, 1, cv2.LINE_AA)

    def draw_crosshair_dot(self, img, cx: int, cy: int, color: Tuple[int, int, int], r: int = 5):
        """Draws reticle crosshair marker."""
        cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r - 4, cy), (cx - r, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx + r, cy), (cx + r + 4, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r - 4), (cx, cy - r), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy + r), (cx, cy + r + 4), color, 1, cv2.LINE_AA)

    def draw_scanline(self, img, frame_idx: int, alpha: float = 0.04):
        """Draws subtle horizontal scanline sweep across the video feed."""
        h = img.shape[0]
        y = int((frame_idx * 4) % h)
        overlay = img.copy()
        cv2.line(overlay, (0, y), (img.shape[1], y), self.white, 1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_person(self, frame, person: Dict[str, Any]):
        x1, y1, x2, y2 = person["box"]
        conf = person["conf"]

        self.alpha_rect(frame, x1, max(0, y1 - 18), x2, y1, self.accent_cyan, 0.65)
        self.corner_box(frame, x1, y1, x2, y2, self.accent_cyan, 2, 16)
        cv2.putText(frame, f"PERSON  {conf:.0%}", (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.dark_bg, 1, cv2.LINE_AA)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.draw_crosshair_dot(frame, cx, cy, self.accent_cyan, 5)

        # Draw detected wrist joints
        for (hx, hy) in person.get("wrists", []):
            cv2.circle(frame, (hx, hy), 4, self.green_safe, -1, cv2.LINE_AA)
            cv2.circle(frame, (hx, hy), 7, self.accent_cyan, 1, cv2.LINE_AA)

    def draw_hand_link(self, frame, hand_pt: Tuple[int, int], waste_box: List[int]):
        """Draws laser guideline connecting wrist keypoint to held object."""
        wcx = (waste_box[0] + waste_box[2]) // 2
        wcy = (waste_box[1] + waste_box[3]) // 2
        hx, hy = hand_pt
        cv2.line(frame, (hx, hy), (wcx, wcy), self.green_safe, 1, cv2.LINE_AA)
        cv2.circle(frame, (wcx, wcy), 3, self.green_safe, -1, cv2.LINE_AA)

    def draw_track_item(self, frame, track: TrackedItem, frame_idx: int, alert_threshold: int) -> bool:
        """Renders bounding box and status chips for a tracked item. Returns True if confirmed litter."""
        x1, y1, x2, y2 = track.box
        label = track.label

        if track.state == TrackState.CONFIRMED:
            blink = (frame_idx // 8) % 2 == 0  # Fixed emoji bug!
            color = self.red_alert if blink else self.red_bright
            self.alpha_rect(frame, x1, y1, x2, y2, self.red_alert, 0.20)
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), self.red_alert, 1)
            self.corner_box(frame, x1, y1, x2, y2, color, 3, 18)
            self.label_chip(frame, f"LITTER  #{track.track_id:02d}", x1, y1, self.red_alert, 0.85, scale=0.44)
            cv2.putText(frame, label, (x1 + 4, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 220), 1, cv2.LINE_AA)
            return True

        elif track.state == TrackState.UNATTENDED:
            pct = min(100, int((track.unattended_frames / alert_threshold) * 100))
            self.alpha_rect(frame, x1, y1, x2, y2, self.amber_warning, 0.12)
            self.corner_box(frame, x1, y1, x2, y2, self.amber_warning, 2, 12)

            # Progress bar towards confirmed litter alert
            bar_w = x2 - x1
            filled = int(bar_w * (pct / 100.0))
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 5), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 5), self.amber_warning, -1)
            self.label_chip(frame, f"{label} {pct}%  #{track.track_id:02d}", x1, y1, self.amber_warning, 0.75)
            return False

        elif track.state == TrackState.HELD:
            self.alpha_rect(frame, x1, y1, x2, y2, self.green_safe, 0.10)
            self.corner_box(frame, x1, y1, x2, y2, self.green_safe, 1, 10)
            self.label_chip(frame, f"{label} HELD", x1, y1, self.green_safe, 0.70)
            if track.hand_pt is not None:
                self.draw_hand_link(frame, track.hand_pt, track.box)
            return False

        return False

    def draw_sidebar(
        self,
        canvas,
        fw: int,
        fh: int,
        fps: float,
        n_persons: int,
        n_litter: int,
        total_incidents: int,
        tracks: Dict[int, TrackedItem],
        frame_idx: int,
    ):
        px = fw
        pw = self.panel_width

        # Panel Background
        canvas[0:fh, px:px + pw] = np.array(self.panel_bg, dtype=np.uint8)

        # Top accent bar
        cv2.rectangle(canvas, (px, 0), (px + pw, 3), self.accent_cyan, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_m = cv2.FONT_HERSHEY_DUPLEX
        y = 26

        # Title
        cv2.putText(canvas, "LITTER WATCH AI", (px + 12, y), font_m, 0.56, self.accent_cyan, 1, cv2.LINE_AA)
        y += 14
        cv2.line(canvas, (px + 10, y), (px + pw - 10, y), self.panel_line, 1)
        y += 18

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(canvas, ts, (px + 10, y), font, 0.36, (140, 140, 140), 1, cv2.LINE_AA)
        y += 22

        # Telemetry Stats
        def stat_row(label: str, val: Any, col: Tuple[int, int, int] = self.white):
            nonlocal y
            cv2.putText(canvas, label, (px + 12, y), font, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(val), (px + pw - 60, y), font_m, 0.50, col, 1, cv2.LINE_AA)
            y += 22

        cv2.line(canvas, (px + 10, y), (px + pw - 10, y), self.panel_line, 1)
        y += 14
        stat_row("FPS", f"{fps:5.1f}")
        stat_row("PERSONS", n_persons, self.accent_cyan)
        stat_row("LITTER", n_litter, self.red_alert if n_litter > 0 else self.green_safe)
        stat_row("LOGGED", total_incidents, (100, 200, 255) if total_incidents > 0 else (120, 120, 120))

        # Active Alerts Section
        y += 4
        cv2.line(canvas, (px + 10, y), (px + pw - 10, y), self.panel_line, 1)
        y += 14
        cv2.putText(canvas, "ACTIVE ALERTS", (px + 12, y), font, 0.37, (110, 110, 110), 1, cv2.LINE_AA)
        y += 18

        confirmed = [t for t in tracks.values() if t.state == TrackState.CONFIRMED and t.lost == 0]

        if not confirmed:
            cv2.putText(canvas, "  none", (px + 12, y), font, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
            y += 20
        else:
            for t in confirmed[:5]:
                blink = (frame_idx // 10) % 2 == 0
                dot_col = self.red_alert if blink else (80, 80, 80)
                cv2.circle(canvas, (px + 16, y - 4), 4, dot_col, -1, cv2.LINE_AA)
                name = t.label[:16]
                cv2.putText(canvas, f"#{t.track_id:02d}  {name}", (px + 26, y), font, 0.37, (210, 210, 210), 1, cv2.LINE_AA)
                y += 20

        # Legend
        y = fh - 105
        cv2.line(canvas, (px + 10, y), (px + pw - 10, y), self.panel_line, 1)
        y += 14

        def legend_row(col: Tuple[int, int, int], text: str):
            nonlocal y
            cv2.rectangle(canvas, (px + 12, y - 8), (px + 20, y), col, -1)
            cv2.putText(canvas, text, (px + 26, y), font, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
            y += 16

        legend_row(self.accent_cyan, "Person (Pose)")
        legend_row(self.green_safe, "Held (in Hand)")
        legend_row(self.amber_warning, "Unattended Drop")
        legend_row(self.red_alert, "Confirmed Litter")

        # Divider between main frame and sidebar
        cv2.line(canvas, (px, 0), (px, fh), self.panel_line, 1)

    def draw_alert_banner(self, canvas, n_litter: int, frame_idx: int, total_width: int):
        """Top full-width alert banner."""
        if n_litter == 0:
            return
        blink = (frame_idx // 6) % 2 == 0
        bg = self.red_alert if blink else (20, 20, 60)
        self.alpha_rect(canvas, 0, 0, total_width, 36, bg, 0.85)
        msg = f"  WARNING: {n_litter} LITTERING VIOLATION{'S' if n_litter > 1 else ''} DETECTED"
        cv2.putText(canvas, msg, (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, self.white, 1, cv2.LINE_AA)

    def draw_watermark(self, canvas, fh: int, fw: int):
        """Bottom right surveillance timestamp and camera identifier."""
        cv2.putText(canvas, "CAM-01  REC", (fw - 110, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
        blink = int(time.time() * 2) % 2 == 0
        if blink:
            cv2.circle(canvas, (fw - 118, fh - 16), 4, self.red_alert, -1, cv2.LINE_AA)
