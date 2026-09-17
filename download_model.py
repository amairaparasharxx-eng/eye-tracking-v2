#!/usr/bin/env python3
"""Download the MediaPipe face landmark model (about 3 MB).

Only needed if your MediaPipe version does not bundle the older
solutions API. Run once, then the program works offline forever.
"""

import os
import sys
import urllib.request

from omsexam.landmarks import MODEL_FILENAME, MODEL_URL

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
if os.path.isfile(dest):
    print(f"Already present: {dest}")
    sys.exit(0)

print(f"Downloading to {dest} ...")
try:
    urllib.request.urlretrieve(MODEL_URL, dest)
except Exception as exc:
    print(f"Download failed: {exc}")
    print(f"Download it manually from:\n  {MODEL_URL}\nand save it as {dest}")
    sys.exit(1)
print("Done.")
