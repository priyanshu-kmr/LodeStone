"""The image quality gate. Images that fail stay findable by filename only."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import SETTINGS, Settings

FULL = "full"
SMALL = "small"
UNREADABLE = "unreadable"
BLANK = "blank"
BLURRY = "blurry"


@dataclass
class QualityResult:
    reason: str
    sharpness: float | None = None
    image: Image.Image | None = None

    @property
    def passed(self) -> bool:
        return self.reason == FULL


def sharpness_score(gray: np.ndarray, settings: Settings = SETTINGS) -> float:
    """Variance of the Laplacian in the image's sharpest area, at a fixed scale.

    Fixed scale, because a cut-off tuned at one resolution does not carry over to another.
    Sharpest tile, so a sharp subject on a blurred background still passes.
    """
    h, w = gray.shape
    scale = settings.blur_long_side / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    tiles = settings.blur_tiles
    th, tw = lap.shape[0] // tiles, lap.shape[1] // tiles
    if not th or not tw:
        return float(lap.var())
    return max(
        float(lap[r * th : (r + 1) * th, c * tw : (c + 1) * tw].var())
        for r in range(tiles)
        for c in range(tiles)
    )


def _flatten(img: Image.Image) -> Image.Image:
    """Put transparent images on white, so their hidden pixels don't affect the result."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return img.convert("RGB")


def assess(path: str, settings: Settings = SETTINGS, force: bool = False) -> QualityResult:
    """Run the checks cheapest-first. `force` is the UI's 'Index anyway' button."""
    try:
        with Image.open(path) as img:
            # The header gives the size without decoding any pixels.
            w, h = img.size
            if not force and w < settings.min_image_side and h < settings.min_image_side:
                return QualityResult(SMALL)
            # Large JPEGs can decode at reduced size, which is much faster.
            img.draft("RGB", (settings.held_image_long_side * 2, settings.held_image_long_side * 2))
            img = ImageOps.exif_transpose(img)
            rgb = _flatten(img)
    except Exception:  # Pillow raises many types for corrupt or unsupported files.
        return QualityResult(UNREADABLE)

    gray = np.asarray(rgb.convert("L"))
    score = sharpness_score(gray, settings)
    if not force and score < settings.blur_threshold:
        # Low variation alone is not enough to call it blank: a small sharp subject on a plain
        # background also has low variation, but it passes the sharpness check above.
        reason = BLANK if float(gray.std()) < settings.blank_std else BLURRY
        return QualityResult(reason, sharpness=score)

    rgb.thumbnail((settings.held_image_long_side, settings.held_image_long_side))
    return QualityResult(FULL, sharpness=score, image=rgb)
