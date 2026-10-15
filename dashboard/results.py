"""Validate committed assets and compute binary metrics without loading models."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

ASSET_DIR = Path(__file__).resolve().parent


class AssetError(ValueError):
    """A demo asset is missing, damaged, or inconsistent with the manifest."""


def checked_asset(filename: str, directory: Path = ASSET_DIR) -> tuple[Path, str]:
    """Only allow manifest-listed files directly inside the dashboard directory."""
    if Path(filename).name != filename:
        raise AssetError("Expected a dashboard asset filename.")
    try:
        manifest = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
        expected = manifest["assets"][filename]["sha256"]
        path = directory / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise AssetError(
            f"Cannot read {filename} or its manifest. Restore the committed dashboard assets."
        ) from exc
    if digest != expected:
        raise AssetError(
            f"{filename} differs from the saved baseline. Restore it or review and update "
            "the artifact manifest before displaying these results."
        )
    return path, digest


def read_predictions(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AssetError(f"Cannot read predictions from {path.name}.") from exc
    if set(frame.columns) != {"actual", "predicted"} or len(frame.columns) != 2:
        raise AssetError("Predictions must contain exactly the columns actual and predicted.")
    if frame.empty:
        raise AssetError("Predictions contain no rows.")
    for column in ("actual", "predicted"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.isin([0, 1]).all():
            raise AssetError(f"Column {column} must contain only non-missing binary labels 0 or 1.")
        frame[column] = values.astype(int)
    return frame


def read_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return source.copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise AssetError(f"Cannot decode chart {path.name}. Restore the committed PNG.") from exc


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def count(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else None

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.count

    @property
    def negative_baseline_accuracy(self) -> float:
        return (self.tn + self.fp) / self.count


def binary_metrics(frame: pd.DataFrame) -> BinaryMetrics:
    """Summarize labels from read_predictions; class 1 is the positive class."""
    actual = frame["actual"]
    predicted = frame["predicted"]
    return BinaryMetrics(
        tp=int(((actual == 1) & (predicted == 1)).sum()),
        fp=int(((actual == 0) & (predicted == 1)).sum()),
        fn=int(((actual == 1) & (predicted == 0)).sum()),
        tn=int(((actual == 0) & (predicted == 0)).sum()),
    )
