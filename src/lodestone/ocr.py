"""OCR with RapidOCR and PP-OCRv6 small, on the CPU, leaving the GPU free for embeddings."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MIN_LINE_SCORE = 0.7
MIN_NUMBER_SCORE = 0.9


@dataclass
class OcrResult:
    text: str
    score: float | None


class Ocr:
    """Two model sizes: 'small' for photos, and the faster 'tiny' for scanned document pages.

    Printed pages are easy to read. On a real scanned PDF, tiny at a lower resolution took a
    third of the time per page and kept 98% of the words that small found.
    """

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}
        self._lock = threading.Lock()

    def _load(self, size: str):
        if size not in self._engines:
            # Imported here because it loads slowly and search never needs it.
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

            model = ModelType.TINY if size == "tiny" else ModelType.SMALL
            self._engines[size] = RapidOCR(
                params={
                    "Global.log_level": "error",  # Otherwise it warns on every photo without text.
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.ocr_version": OCRVersion.PPOCRV6,
                    "Det.model_type": model,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.model_type": model,
                }
            )
        return self._engines[size]

    def _run(self, image: Image.Image, size: str) -> OcrResult:
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
        with self._lock:
            result = self._load(size)(bgr)
        return clean_lines(result.txts or (), result.scores or ())

    def read(self, image: Image.Image) -> OcrResult:
        """Read a photo or screenshot."""
        return self._run(image, "small")

    def read_page(self, image: Image.Image) -> OcrResult:
        """Read a scanned document page."""
        return self._run(image, "tiny")


def clean_lines(lines: Sequence[str], scores: Sequence[float]) -> OcrResult:
    """Keep real text only. OCR reads stray marks like 'O' or '8' in ordinary photos."""
    kept = [(line, score) for line, score in zip(lines, scores) if _looks_real(line, score)]
    if not kept:
        return OcrResult("", None)
    return OcrResult(
        "\n".join(line for line, _ in kept), float(sum(s for _, s in kept) / len(kept))
    )


def _looks_real(line: str, score: float) -> bool:
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    if letters >= 3:
        return score >= MIN_LINE_SCORE
    # Numbers are easy to misread from texture, so they need more length and confidence.
    return digits >= 4 and score >= MIN_NUMBER_SCORE
