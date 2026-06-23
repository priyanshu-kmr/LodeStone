"""Turn documents into plain text, using OCR only for pages that have no text layer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .config import SETTINGS, Settings
from .ocr import OcrResult
from .office import shared_worker

log = logging.getLogger(__name__)


class OcrReader(Protocol):
    def read(self, image: Image.Image) -> OcrResult: ...
    def read_page(self, image: Image.Image) -> OcrResult: ...
    def release(self) -> None: ...


@dataclass
class Extracted:
    body: str = ""
    ocr: str = ""
    ocr_score: float | None = None


def read_text_file(path: str, settings: Settings = SETTINGS) -> str:
    with open(path, "rb") as handle:
        raw = handle.read(settings.max_text_bytes)
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def extract_pdf(path: str, ocr: OcrReader, settings: Settings = SETTINGS) -> Extracted:
    import pymupdf

    body: list[str] = []
    ocr_text: list[str] = []
    scores: list[float] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("password-protected PDF")
        ocr_budget = settings.max_pdf_ocr_pages
        collected = 0
        # Only this much can become chunks, so a very long book needn't be read to the end.
        needed = settings.max_chunks_per_file * settings.chunk_chars
        for page in doc:
            if collected >= needed:
                break
            text = page.get_text().strip()
            collected += len(text)
            if len(text) >= settings.pdf_min_page_chars:
                body.append(text)
                continue
            # Almost no text layer: this page is probably a scan. OCR is by far the slowest step,
            # so it stops once this file has enough text to embed.
            if ocr_budget > 0 and collected < settings.embed_char_limit and page.get_images():
                ocr_budget -= 1
                pix = page.get_pixmap(dpi=settings.pdf_ocr_dpi, colorspace=pymupdf.csRGB, alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                result = ocr.read_page(image)
                if result.text:
                    ocr_text.append(result.text)
                    collected += len(result.text)
                    if result.score is not None:
                        scores.append(result.score)
            elif text:
                body.append(text)
    return Extracted(
        "\n\n".join(body),
        "\n\n".join(ocr_text),
        sum(scores) / len(scores) if scores else None,
    )


def extract(path: str, ocr: OcrReader, settings: Settings = SETTINGS) -> Extracted:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path, ocr, settings)
    if ext in {".docx", ".pptx", ".xlsx", ".html", ".htm"}:
        return Extracted(body=shared_worker().convert(path))
    return Extracted(body=read_text_file(path, settings))


def within_embed_limit(texts: list[str], settings: Settings = SETTINGS) -> list[str]:
    """The opening of a file's chunks that fits the embedding limit, cutting the last one to fit."""
    remaining = settings.embed_char_limit
    kept: list[str] = []
    for text in texts:
        if remaining <= 0:
            break
        piece = text[:remaining]
        kept.append(piece)
        remaining -= len(piece)
    return kept


def chunk_text(text: str, settings: Settings = SETTINGS) -> list[str]:
    """Split into overlapping pieces, preferring to break between paragraphs and sentences.

    A whole long file would blur into one vector. Small pieces let one relevant page rank well.
    """
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not text:
        return []
    size, overlap = settings.chunk_chars, settings.chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < settings.max_chunks_per_file:
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(separator, size // 2)
                if cut != -1:
                    end = start + cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
