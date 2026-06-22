"""Folder rules, the dataset check and the file walk."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .config import (
    DOCUMENT_EXTS,
    IMAGE_EXTS,
    NAME_ONLY_EXTS,
    SETTINGS,
    SKIP_DIRS,
    TEMP_PREFIXES,
    TEMP_SUFFIXES,
    Settings,
    is_under,
    norm_path,
)

# Cloud-only files: opening one would download it.
_CLOUD_ONLY = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | 0x00040000  # FILE_ATTRIBUTE_RECALL_ON_OPEN
    | 0x00400000  # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)
_NUMBERED_NAME = re.compile(r"^(?:img|image|frame|sample|dsc|pic)?[_\- ]?\d+", re.IGNORECASE)
_DATASET_DIR_NAMES = {
    "train", "training", "val", "valid", "validation", "test", "testing", "dataset", "datasets",
    "labels", "annotations",
}
_DATASET_FILE_NAMES = {
    "labels.csv", "annotations.csv", "annotations.json", "train.csv", "val.csv", "test.csv",
    "classes.txt", "data.yaml",
}
_CODE_MARKERS = {".git", "src", "requirements.txt", "pyproject.toml", "setup.py"}


def kind_of(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in DOCUMENT_EXTS:
        return "document"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in NAME_ONLY_EXTS:
        return "name_only"
    return None


def is_temp_file(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(TEMP_PREFIXES) or lower.endswith(TEMP_SUFFIXES)


def is_skipped_dir_name(name: str) -> bool:
    lower = name.lower()
    return lower in SKIP_DIRS or lower.endswith(".noindex")


def is_cloud_only(attributes: int) -> bool:
    return bool(attributes & _CLOUD_ONLY)


def path_allowed(path: str, roots: Iterable[str], excludes: Iterable[str]) -> bool:
    """Apply the folder rules to one path, for events from the watcher."""
    path = norm_path(path)
    if is_temp_file(os.path.basename(path)):
        return False
    if any(is_under(path, e) for e in excludes):
        return False
    for root in roots:
        if is_under(path, root):
            relative = os.path.relpath(os.path.dirname(path), root)
            parts = [] if relative == "." else relative.split(os.sep)
            return not any(is_skipped_dir_name(p) for p in parts)
    return False


@dataclass
class FileEntry:
    path: str
    size: int
    mtime_ns: int
    kind: str


@dataclass
class _DirStats:
    path: str
    parent: "_DirStats | None"
    children: list["_DirStats"] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    images: int = 0
    numbered: int = 0
    image_bytes: int = 0
    subtree_images: int = 0
    subtree_numbered: int = 0
    subtree_bytes: int = 0


@dataclass
class DatasetCandidate:
    path: str
    image_count: int
    signs: list[str]


@dataclass
class ScanResult:
    root: str
    files: list[FileEntry]
    candidates: list[DatasetCandidate]


def walk(
    root: str,
    excludes: Iterable[str] = (),
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[FileEntry], _DirStats]:
    """Read file details only, never contents, so this stays light on a whole folder tree."""
    root = norm_path(root)
    excludes = [norm_path(e) for e in excludes]
    top = _DirStats(root, None)
    files: list[FileEntry] = []
    stack = [top]
    while stack:
        if should_stop and should_stop():
            break
        current = stack.pop()
        try:
            entries = list(os.scandir(current.path))
        except OSError:
            continue
        names = {e.name.lower() for e in entries}
        current.names = names
        if "pyvenv.cfg" in names and current is not top:
            continue
        for entry in entries:
            try:
                if entry.is_junction() or entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if is_skipped_dir_name(entry.name):
                        continue
                    if any(is_under(entry.path, e) for e in excludes):
                        continue
                    child = _DirStats(entry.path, current)
                    current.children.append(child)
                    stack.append(child)
                    continue
                kind = kind_of(entry.name)
                if kind is None or is_temp_file(entry.name):
                    continue
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if is_cloud_only(getattr(info, "st_file_attributes", 0)):
                continue
            files.append(FileEntry(norm_path(entry.path), info.st_size, info.st_mtime_ns, kind))
            if kind == "image":
                current.images += 1
                current.image_bytes += info.st_size
                if _NUMBERED_NAME.match(os.path.splitext(entry.name)[0]):
                    current.numbered += 1
    _roll_up(top)
    return files, top


def _roll_up(node: _DirStats) -> None:
    # Iterative, because Windows folder trees can be deep.
    order: list[_DirStats] = []
    stack = [node]
    while stack:
        current = stack.pop()
        order.append(current)
        stack.extend(current.children)
    for current in reversed(order):
        current.subtree_images = current.images + sum(c.subtree_images for c in current.children)
        current.subtree_numbered = current.numbered + sum(c.subtree_numbered for c in current.children)
        current.subtree_bytes = current.image_bytes + sum(c.subtree_bytes for c in current.children)


def _subtree_names(node: _DirStats) -> set[str]:
    names: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        names |= current.names
        stack.extend(current.children)
    return names


def _signs(node: _DirStats, settings: Settings) -> list[str]:
    signs: list[str] = []
    if node.subtree_numbered / node.subtree_images >= settings.dataset_numbered_ratio:
        signs.append("numbered filenames")
    average = node.subtree_bytes / node.subtree_images
    if average < settings.dataset_small_avg_bytes:
        signs.append(f"very small files (about {average / 1024:.0f} KB each)")

    # Nearby code: the folder itself or up to two folders above it.
    current: _DirStats | None = node
    for _ in range(3):
        if current is None:
            break
        if current.names & _CODE_MARKERS or any(n.endswith(".py") for n in current.names):
            signs.append("code nearby")
            break
        current = current.parent

    inside = _subtree_names(node) | (node.parent.names if node.parent else set())
    if inside & _DATASET_DIR_NAMES or inside & _DATASET_FILE_NAMES:
        signs.append("dataset-style names")
    return signs


def find_datasets(top: _DirStats, settings: Settings = SETTINGS) -> list[DatasetCandidate]:
    """Find folders that look like datasets, without ever flagging the watched folder itself.

    Walking down from the root, each step narrows to a child holding almost all of the images.
    That points at the real dataset folder rather than a parent that merely contains it.
    """
    found: list[DatasetCandidate] = []
    stack = list(top.children)
    while stack:
        node = stack.pop()
        if node.subtree_images < settings.dataset_min_images:
            continue
        dominant = max(node.children, key=lambda c: c.subtree_images, default=None)
        if dominant is not None and dominant.subtree_images >= 0.9 * node.subtree_images:
            stack.append(dominant)
            continue
        signs = _signs(node, settings)
        if len(signs) >= settings.dataset_min_signs:
            found.append(DatasetCandidate(node.path, node.subtree_images, signs))
        else:
            stack.extend(node.children)
    return found


def scan_root(
    root: str,
    excludes: Iterable[str] = (),
    settings: Settings = SETTINGS,
    should_stop: Callable[[], bool] | None = None,
) -> ScanResult:
    files, top = walk(root, excludes, should_stop)
    return ScanResult(norm_path(root), files, find_datasets(top, settings))


@dataclass
class FolderEstimate:
    documents: int
    images: int
    is_drive_root: bool

    @property
    def total(self) -> int:
        return self.documents + self.images


def estimate_folder(path: str, excludes: Iterable[str] = ()) -> FolderEstimate:
    """Count files before indexing, so the UI can warn about very large folders."""
    files, _ = walk(path, excludes)
    images = sum(f.kind == "image" for f in files)
    resolved = Path(norm_path(path))
    return FolderEstimate(len(files) - images, images, resolved == Path(resolved.anchor))
