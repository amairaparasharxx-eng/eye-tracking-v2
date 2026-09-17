#!/usr/bin/env python3
"""Eye Sign Recorder - entry point.

THIS IS NOT A MEDICAL DEVICE. It cannot diagnose myasthenia gravis or any
other condition. See DISCLAIMER.txt.

Usage:
    python main.py              # normal, windowed
    python main.py --check      # verify the installation without a camera
"""

import sys


def check() -> int:
    ok = True
    print("Checking installation...\n")
    for name, hint in (
        ("cv2", "pip install opencv-python"),
        ("numpy", "pip install numpy"),
        ("mediapipe", "pip install mediapipe"),
        ("PIL", "pip install pillow"),
    ):
        try:
            __import__(name)
            print(f"  [ ok ] {name}")
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {name}: {exc}\n         fix with: {hint}")

    try:
        import tkinter  # noqa: F401
        print("  [ ok ] tkinter")
    except Exception:
        ok = False
        print("  [FAIL] tkinter is missing.")
        print("         Linux:  sudo apt install python3-tk")
        print("         macOS:  install Python from python.org, not Homebrew's minimal build")

    try:
        import pyttsx3  # noqa: F401
        print("  [ ok ] pyttsx3 (spoken instructions available)")
    except Exception:
        print("  [note] pyttsx3 not installed - spoken instructions will be off")

    from omsexam.landmarks import FaceTracker
    model = FaceTracker._find_model()
    if model:
        print(f"  [ ok ] face model found at {model}")
    else:
        print("  [note] face model not found - run: python download_model.py")
        print("         (only needed if your mediapipe lacks the legacy solutions API)")

    try:
        from omsexam import protocols
        print(f"  [ ok ] {len(protocols.PROTOCOLS)} protocols loaded")
    except Exception as exc:
        ok = False
        print(f"  [FAIL] protocols: {exc}")

    print("\nReady." if ok else "\nFix the failures above before running.")
    return 0 if ok else 1


def main() -> int:
    if "--check" in sys.argv:
        return check()
    from omsexam.ui import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
