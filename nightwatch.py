"""
NightWatch - Unified Runner (Webcam Enabled)
"""

import argparse
import subprocess
import sys
import os

ENGINE_PATH = os.path.join("scripts", "nightwatch_stage7.py")

def main():
    parser = argparse.ArgumentParser(description="NightWatch Runner")

    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file. If omitted, file picker opens.")

    parser.add_argument("--no-audio", action="store_true",
                        help="Disable alarm sound")

    parser.add_argument("--webcam", type=int, default=None,
                        help="Select webcam index (0,1,2...)")

    args = parser.parse_args()

    cmd = [sys.executable, ENGINE_PATH]

    if args.source:
        cmd += ["--source", args.source]

    if args.no_audio:
        cmd += ["--no-audio"]

    if args.webcam is not None:
        cmd += ["--webcam", str(args.webcam)]

    print("Launching NightWatch engine...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
