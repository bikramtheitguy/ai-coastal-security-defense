"""Publish bounded synthetic UI previews to CI logs for remote visual inspection."""

import base64
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
directory = root / "artifacts/browser"
for name in (
    "replay-desktop.jpg",
    "review-desktop.jpg",
    "replay-mobile.jpg",
    "replay-chart.jpg",
    "replay-mobile-chart.jpg",
    "spatial-research-desktop.jpg",
    "evaluation-desktop.jpg",
    "evaluation-comparison.jpg",
    "evaluation-mobile.jpg",
    "evaluation-mobile-chart.jpg",
    "watch-india-desktop.jpg",
    "watch-odisha-map.jpg",
    "watch-mobile.jpg",
    "watch-mobile-map.jpg",
):
    path = directory / name
    if path.exists() and path.stat().st_size <= 250_000:
        print(
            "SCREENSHOT_JSON="
            + json.dumps(
                {
                    "name": name,
                    "mime_type": "image/jpeg",
                    "base64": base64.b64encode(path.read_bytes()).decode(),
                }
            )
        )
# The initial lock proposal is read back verbatim before being committed for subsequent runs.
lock = root / "browser_tests/requirements.txt"
if lock.exists():
    print("BROWSER_LOCK_JSON=" + json.dumps(lock.read_text(encoding="utf-8")))
