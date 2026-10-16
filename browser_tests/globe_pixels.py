"""Inspect a CI canvas screenshot for rendered land and an amber vessel glyph."""

import json
import sys

from PIL import Image

with Image.open(sys.argv[1]) as source:
    image = source.convert("RGB")
    land = 0
    amber = []
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = image.getpixel((x, y))
            if 25 < r < 85 and 45 < g < 120 and g > r + 15 and g > b:
                land += 1
            if r > 240 and 170 < g < 220 and 70 < b < 140:
                amber.append((x, y))
    print(
        json.dumps(
            {
                "land_pixels": land,
                "amber_pixels": len(amber),
                "amber_center": [sum(p[i] for p in amber) / len(amber) for i in (0, 1)]
                if amber
                else None,
            }
        )
    )
