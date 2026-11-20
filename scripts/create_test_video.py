"""Generates a brief synthetic video for smoke-testing the end-to-end pipeline without a webcam."""

from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_video(output_path="demo_sample.mp4", num_frames=60, width=640, height=480):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), (25, 25, 28), dtype=np.uint8)

        # Draw a simulated floor line
        cv2.line(frame, (0, int(height * 0.75)), (width, int(height * 0.75)), (60, 60, 60), 2)

        # Draw a moving white circle (simulating an object)
        cx = int(width * 0.3 + (i * 3))
        cy = int(height * 0.4 + (i * 2))
        cv2.circle(frame, (cx, cy), 15, (200, 200, 200), -1)

        cv2.putText(
            frame,
            f"Test Frame {i + 1}/{num_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"[OK] Generated synthetic video: {output_path} ({num_frames} frames)")


if __name__ == "__main__":
    generate_synthetic_video()
