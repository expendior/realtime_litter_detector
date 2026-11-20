"""Real-Time Litter Watch AI: Intelligent Surveillance & Evidence Logging System.

USAGE EXAMPLES:
    python main.py                          # Default webcam (index 0)
    python main.py --source 1               # External webcam (index 1)
    python main.py --source footage.mp4     # Video file analysis
    python main.py --conf 0.35              # Custom waste detection threshold
    python main.py --no-record              # Disable auto incident clip recording
"""

import argparse
import sys
from pathlib import Path
import yaml

from src.pipeline import LitterDetectionPipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[WARNING] Config file '{config_path}' not found. Using default configurations.")
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Litter Watch AI: Real-Time Littering Detection & Evidence Vault",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="1",
        help="Input video source: Camera index (0, 1), video file path (.mp4, .avi), or RTSP URL",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Waste detection confidence threshold (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic 5-second evidence video clip export upon violation",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for servers, Docker, or batch processing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export the full annotated surveillance feed (.mp4)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="Run neural nets every N frames (2 or 3 boosts FPS on CPU; 1 runs every frame)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (default 640; higher like 768 detects farther objects)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine whether source is camera index or file/URL
    source = int(args.source) if args.source.isdigit() else args.source

    config = load_config(args.config)

    print("═" * 70)
    print(" LITTER WATCH AI  ·  Surveillance & Event Telemetry System")
    print("═" * 70)
    print(f" Source          : {source}")
    print(f" Auto-Record     : {'Disabled' if args.no_record else 'Enabled (incidents/)'}")
    print(f" Config File     : {args.config}")
    print("═" * 70)

    try:
        pipeline = LitterDetectionPipeline(
            config=config,
            waste_conf_override=args.conf,
            enable_record=not args.no_record,
            headless=args.headless,
            output_path=args.output,
            skip_override=args.skip,
            imgsz_override=args.imgsz,
        )
        pipeline.run(source=source)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted execution. Exiting...")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
