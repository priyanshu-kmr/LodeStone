"""Paths, file types and tuning values, in one place so they are easy to explain."""

from __future__ import annotations

import ctypes
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
EMBEDDING_DIM = 768

DOCUMENT_EXTS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv", ".log", ".html", ".htm",
}
# Formats we cannot read the contents of, so they stay findable by filename only.
NAME_ONLY_EXTS = {".doc", ".ppt", ".xls", ".rtf"}
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic",
}

SKIP_DIRS = {
    "node_modules", ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "site-packages",
    ".cache", ".idea", ".vscode", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
    "appdata", "$recycle.bin", "system volume information", "build", "dist", "target",
}
TEMP_PREFIXES = ("~$", ".~")
TEMP_SUFFIXES = (".tmp", ".temp", ".crdownload", ".part", ".partial", ".download")

# The range offered in Settings for embed_token_limit.
TOKEN_LIMIT_MAX = 5000
TOKEN_LIMIT_STEP = 100

# Personal project datasets, relative to the home folder. Seeded once on first run and
# editable in Settings afterwards.
SEED_EXCLUDES = (
    r"Desktop\codes\ML\human-wildlife-conflict\Elephant",
    r"Desktop\Btech\amazon_ml\student_resource3\images",
)


@dataclass(frozen=True)
class Settings:
    # Quality gate. The blur cut-off comes from measuring the user's own folders and is
    # deliberately low, because a wrongly flagged image loses its content search.
    min_image_side: int = 200
    blank_std: float = 5.0
    blur_threshold: float = 25.0
    blur_long_side: int = 512
    blur_tiles: int = 4

    # Dataset check: 1,000+ images plus at least two other signs.
    dataset_min_images: int = 1000
    dataset_min_signs: int = 2
    dataset_small_avg_bytes: int = 50 * 1024
    dataset_numbered_ratio: float = 0.8
    large_folder_warning: int = 5000

    # Ingestion.
    debounce_seconds: float = 5.0
    delete_delay_seconds: float = 10.0
    rescan_hours: float = 6.0
    # A file another program keeps writing to, like a log, would otherwise be re-embedded
    # every few seconds and stop the app from ever going idle.
    reindex_cooldown_seconds: float = 600.0
    # Large PDFs and text files are fine, because only their opening part gets read. Office files
    # have to be unpacked in full, so they get a much lower limit.
    max_document_bytes: int = 1024 * 1024 * 1024
    max_office_bytes: int = 100 * 1024 * 1024
    office_timeout_seconds: float = 120.0
    sheet_max_rows: int = 1000
    sheet_max_chars: int = 400_000
    max_text_bytes: int = 5 * 1024 * 1024
    max_image_bytes: int = 100 * 1024 * 1024
    max_pdf_ocr_pages: int = 50
    pdf_min_page_chars: int = 40
    pdf_ocr_dpi: int = 130
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    max_chunks_per_file: int = 300
    # Very short text gets a generic vector that scores highly against everything, so it is
    # kept for keyword search only.
    min_embed_chars: int = 8
    # Only the start of each file gets embedded, and OCR stops once a scanned PDF has this much
    # text. Everything already read stays in keyword search, so exact words can still be found.
    embed_token_limit: int = 500
    chars_per_token: int = 4  # A common rough estimate for English text.

    # Embedding. Measured on an RTX 3060: at 280 tokens, batches above 4 add VRAM but no speed.
    image_tokens: int = 280
    image_batch_size: int = 4
    text_batch_size: int = 16
    # Long unbroken text like hex dumps becomes huge token counts, which is slow and fills the GPU.
    max_text_tokens: int = 512
    # Stay within this share of VRAM, and leave the rest for everything else on the laptop.
    max_vram_fraction: float = 0.65
    held_image_long_side: int = 1280
    vision_idle_unload_seconds: float = 60.0
    # The next search after this takes a few seconds, because the model has to load again.
    search_idle_unload_seconds: float = 600.0
    cpu_threads: int = 4

    # Search.
    index_min_vectors: int = 10_000
    rrf_k: int = 60
    candidates_per_list: int = 60
    default_top_k: int = 20
    # Stops a list with almost no spread from making tiny differences look important.
    min_score_spread: float = 0.02
    # A keyword match must cover at least this share of the query's words to count.
    keyword_min_coverage: float = 0.5

    @property
    def embed_char_limit(self) -> int:
        return self.embed_token_limit * self.chars_per_token


SETTINGS = Settings()


@dataclass(frozen=True)
class Paths:
    root: Path
    catalog: Path = field(init=False)
    vectors: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog", self.root / "catalog.db")
        object.__setattr__(self, "vectors", self.root / "vectors.lance")


def data_dir() -> Path:
    """The index lives in LOCALAPPDATA because it is large and tied to this machine."""
    override = os.environ.get("LODESTONE_DATA")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) if local else Path.home() / "AppData" / "Local") / "LodeStone"


def get_paths() -> Paths:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return Paths(root)


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


_KNOWN_FOLDERS = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
}


def _known_folder(name: str) -> Path:
    """Ask Windows, because these folders can be redirected elsewhere, such as OneDrive."""
    try:
        guid = _GUID.from_string(_KNOWN_FOLDERS[name])
        out = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(out)
        )
        if result == 0 and out.value:
            path = Path(out.value)
            ctypes.windll.ole32.CoTaskMemFree(out)
            return path
    except (AttributeError, OSError):
        pass
    return Path.home() / name


def default_folders() -> list[Path]:
    return [p for p in (_known_folder(n) for n in _KNOWN_FOLDERS) if p.is_dir()]


def seed_excludes() -> list[Path]:
    home = Path.home()
    return [home / rel for rel in SEED_EXCLUDES if (home / rel).is_dir()]


def norm_path(path: str | os.PathLike[str]) -> str:
    """One spelling per path, so the same file can never be stored twice."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def is_under(path: str, folder: str) -> bool:
    """True if path is the folder or inside it, ignoring case as Windows does."""
    p = os.path.normcase(path)
    f = os.path.normcase(folder).rstrip("\\/")
    return p == f or p.startswith(f + os.sep)
