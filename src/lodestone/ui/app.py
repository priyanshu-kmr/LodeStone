"""The search window, settings, and the system tray. All heavy work runs off the UI thread."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import winreg
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QImageReader, QPainter, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import data_dir
from ..engine import Engine
from ..search import ALL, DOCUMENTS, IMAGES, SearchResult

log = logging.getLogger(__name__)

THUMB = 56
SEARCH_DELAY_MS = 220
SKIP_REASONS = {
    "small": "Too small",
    "blurry": "Very blurry",
    "blank": "Blank",
    "unreadable": "Won't open",
}
AUTOSTART_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
AUTOSTART_NAME = "LodeStone"

STYLE = """
QMainWindow, QDialog { background: palette(window); }
#search { font-size: 17px; padding: 10px 12px; border: 1px solid palette(mid); border-radius: 8px; }
#search:focus { border: 1px solid palette(highlight); }
#results { border: none; outline: none; background: transparent; }
#results::item { border-radius: 6px; margin: 1px 0; }
#results::item:selected { background: palette(highlight); color: palette(highlighted-text); }
#resultName { font-weight: 600; font-size: 14px; }
#resultPath, #resultSnippet, #statusText { color: palette(placeholder-text); }
#resultPath, #resultSnippet { font-size: 12px; }
#filter { padding: 4px 12px; border: 1px solid palette(mid); border-radius: 12px; background: transparent; }
#filter:checked { background: palette(highlight); color: palette(highlighted-text); border-color: palette(highlight); }
#promptBanner { background: palette(alternate-base); border: 1px solid palette(mid); border-radius: 8px; }
#emptyTitle { font-size: 16px; font-weight: 600; }
"""


class _Bridge(QObject):
    """Lives on the UI thread, so results from worker threads always arrive back there."""

    deliver = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.deliver.connect(self._call, Qt.ConnectionType.QueuedConnection)

    def _call(self, done: Callable[[object], None], result: object) -> None:
        done(result)


_bridge: _Bridge | None = None


def _run(pool: QThreadPool, work: Callable[[], object], done: Callable[[object], None] | None = None) -> None:
    """Run work on a pool thread, then call done with the result. Call this on the UI thread."""
    global _bridge
    if _bridge is None:
        _bridge = _Bridge()
    pool.start(_Task(work, done, _bridge))


class _Task(QRunnable):
    def __init__(self, work: Callable[[], object], done: Callable[[object], None] | None, bridge: _Bridge) -> None:
        super().__init__()
        self._work = work
        self._done = done
        self._bridge = bridge

    def run(self) -> None:
        try:
            result = self._work()
        except Exception as error:
            log.exception("background task failed")
            result = error
        if self._done:
            self._bridge.deliver.emit(self._done, result)


def app_icon() -> QIcon:
    """A simple drawn icon, so the app needs no image files."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#2f5d8c"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(2, 2, 60, 60, 14, 14)
    painter.setPen(QColor("white"))
    font = QFont()
    font.setPixelSize(36)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "L")
    painter.end()
    return QIcon(pixmap)


def reveal_in_explorer(path: str) -> None:
    subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])


def load_thumbnail(path: str, size: int = THUMB):
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    source = reader.size()
    if source.isValid():
        reader.setScaledSize(source.scaled(QSize(size, size), Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    return None if image.isNull() else image


class ElidedLabel(QLabel):
    """One line of text that shortens with an ellipsis instead of being cut off at the edge."""

    def __init__(self, text: str, mode: Qt.TextElideMode) -> None:
        super().__init__(text)
        self._full = text
        self._mode = mode
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setMinimumWidth(1)
        self.setToolTip(text)

    def resizeEvent(self, event) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, self._mode, self.width()))
        super().resizeEvent(event)


class ResultRow(QWidget):
    def __init__(self, result: SearchResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB, THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setText("IMG" if result.kind == "image" else Path(result.name).suffix[1:].upper()[:4])
        self.thumb.setStyleSheet("border: 1px solid palette(mid); border-radius: 6px;")

        name = ElidedLabel(result.name, Qt.TextElideMode.ElideMiddle)
        name.setObjectName("resultName")
        # The middle of a path is the least useful part, so that is what gets shortened.
        folder = ElidedLabel(str(Path(result.path).parent), Qt.TextElideMode.ElideMiddle)
        folder.setObjectName("resultPath")
        snippet = ElidedLabel(result.snippet, Qt.TextElideMode.ElideRight)
        snippet.setObjectName("resultSnippet")
        # Never call show or setVisible(True) here: without a parent yet, it would open as a window.
        if not result.snippet:
            snippet.hide()

        column = QVBoxLayout()
        column.setSpacing(1)
        column.addStretch(1)
        column.addWidget(name)
        column.addWidget(folder)
        column.addWidget(snippet)
        column.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)
        layout.addWidget(self.thumb)
        layout.addLayout(column, 1)

    def set_thumbnail(self, image) -> None:
        self.thumb.setText("")
        self.thumb.setPixmap(QPixmap.fromImage(image))


class SearchWindow(QMainWindow):
    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self._request = 0
        self._settings: SettingsDialog | None = None
        self._waiting_for_model = False
        self.quitting = False

        self.setWindowTitle("LodeStone")
        self.setWindowIcon(app_icon())
        self.resize(760, 620)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("search")
        self.search_box.setPlaceholderText("Describe what you're looking for")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setAccessibleName("Search")
        self.search_box.textChanged.connect(self._text_changed)
        self.search_box.returnPressed.connect(self._search_now)
        self.search_box.installEventFilter(self)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(SEARCH_DELAY_MS)
        self.timer.timeout.connect(self._search_now)

        self.filters = QButtonGroup(self)
        self.filters.setExclusive(True)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        for label, mode in (("All", ALL), ("Documents", DOCUMENTS), ("Images", IMAGES)):
            button = QPushButton(label)
            button.setObjectName("filter")
            button.setCheckable(True)
            button.setProperty("mode", mode)
            button.setChecked(mode == ALL)
            self.filters.addButton(button)
            filter_row.addWidget(button)
        self.filters.buttonClicked.connect(lambda _: self._search_now())
        filter_row.addStretch(1)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        filter_row.addWidget(settings_button)

        self.prompt_banner = QFrame()
        self.prompt_banner.setObjectName("promptBanner")
        banner_layout = QHBoxLayout(self.prompt_banner)
        self.prompt_text = QLabel()
        self.prompt_text.setWordWrap(True)
        self.prompt_text.setTextFormat(Qt.TextFormat.PlainText)
        skip = QPushButton("Skip this folder")
        skip.clicked.connect(lambda: self._decide_prompt(False))
        index = QPushButton("Index anyway")
        index.clicked.connect(lambda: self._decide_prompt(True))
        banner_layout.addWidget(self.prompt_text, 1)
        banner_layout.addWidget(skip)
        banner_layout.addWidget(index)
        self.prompt_banner.hide()
        self._prompt_path: str | None = None

        self.results = QListWidget()
        self.results.setObjectName("results")
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results.customContextMenuRequested.connect(self._context_menu)
        self.results.itemActivated.connect(self._open_item)
        self.results.hide()

        self.empty_title = QLabel("Search your files by describing them")
        self.empty_title.setObjectName("emptyTitle")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_body = QLabel('Try "my interview notes" or "image of an owl next to a lamp".')
        self.empty_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_body.setWordWrap(True)
        self.empty_body.setObjectName("statusText")
        self.empty = QWidget()
        empty_layout = QVBoxLayout(self.empty)
        empty_layout.addStretch(1)
        empty_layout.addWidget(self.empty_title)
        empty_layout.addWidget(self.empty_body)
        empty_layout.addStretch(2)

        self.status_text = QLabel()
        self.status_text.setObjectName("statusText")
        self.status_text.setTextFormat(Qt.TextFormat.PlainText)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)
        layout.addWidget(self.search_box)
        layout.addLayout(filter_row)
        layout.addWidget(self.prompt_banner)
        layout.addWidget(self.results, 1)
        layout.addWidget(self.empty, 1)
        layout.addWidget(self.status_text)
        self.setCentralWidget(central)

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start()
        self.refresh_status()
        self.search_box.setFocus()

    # ---- search -------------------------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:
        # Arrow keys move through results without leaving the search box.
        if watched is self.search_box and event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.results.count():
                step = 1 if event.key() == Qt.Key.Key_Down else -1
                row = max(0, min(self.results.currentRow() + step, self.results.count() - 1))
                self.results.setCurrentRow(row)
                return True
            if event.key() == Qt.Key.Key_Return and self.results.currentItem() and not self.timer.isActive():
                self._open_item(self.results.currentItem())
                return True
        return super().eventFilter(watched, event)

    def _mode(self) -> str:
        return self.filters.checkedButton().property("mode")

    def _text_changed(self, text: str) -> None:
        if text.strip():
            self.timer.start()
        else:
            self.timer.stop()
            self._request += 1
            self.results.clear()
            self._show_empty("Search your files by describing them",
                             'Try "my interview notes" or "image of an owl next to a lamp".')

    def _search_now(self) -> None:
        self.timer.stop()
        query = self.search_box.text().strip()
        if not query:
            return
        self._request += 1
        request, mode = self._request, self._mode()
        if not self.engine.search_model_loaded:
            self._waiting_for_model = True
            self.status_text.setText("Loading the search model, which takes a few seconds...")
        _run(
            self.pool,
            lambda: self.engine.search(query, mode=mode),
            lambda results: self._show_results(request, query, results),
        )

    def _show_results(self, request: int, query: str, results) -> None:
        if request != self._request:
            return  # A newer search has started, so this answer is out of date.
        self._waiting_for_model = False
        self.results.clear()
        if isinstance(results, Exception):
            self._show_empty("Search failed", str(results))
            return
        if not results:
            self._show_empty(f'Nothing found for "{query}"', "Try describing it differently, or check which folders are indexed in Settings.")
            return
        self.empty.hide()
        self.results.show()
        for result in results:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, result.path)
            row = ResultRow(result)
            item.setSizeHint(QSize(0, THUMB + 14))
            self.results.addItem(item)
            self.results.setItemWidget(item, row)
            if result.kind == "image":
                self._load_thumbnail(request, result.path, row)
        self.results.setCurrentRow(0)

    def _load_thumbnail(self, request: int, path: str, row: ResultRow) -> None:
        def show(image) -> None:
            if request == self._request and image is not None and not isinstance(image, Exception):
                row.set_thumbnail(image)

        _run(self.pool, lambda: load_thumbnail(path), show)

    def _show_empty(self, title: str, body: str) -> None:
        self.results.hide()
        self.empty_title.setText(title)
        self.empty_body.setText(body)
        self.empty.show()

    # ---- results ------------------------------------------------------------------------------

    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError as error:
            QMessageBox.warning(self, "Couldn't open the file", f"{path}\n\n{error}")

    def _context_menu(self, position) -> None:
        item = self.results.itemAt(position)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Open", lambda: self._open_item(item))
        menu.addAction("Show in folder", lambda: reveal_in_explorer(path))
        menu.addAction("Copy path", lambda: QGuiApplication.clipboard().setText(path))
        menu.exec(self.results.viewport().mapToGlobal(position))

    # ---- status and prompts -------------------------------------------------------------------

    def refresh_status(self) -> None:
        _run(self.pool, self.engine.status, self._apply_status)

    def _apply_status(self, status) -> None:
        if isinstance(status, Exception):
            return
        parts = [status.state, f"{status.indexed:,} files indexed"]
        if status.queued:
            parts.append(f"{status.queued:,} to go")
        if status.errors:
            parts.append(f"{status.errors:,} couldn't be read")
        if not self._waiting_for_model:
            self.status_text.setText(" · ".join(parts))
        if status.pending_prompts:
            _run(self.pool, self.engine.dataset_prompts, self._apply_prompts)
        else:
            self.prompt_banner.hide()
            self._prompt_path = None

    def _apply_prompts(self, prompts) -> None:
        if isinstance(prompts, Exception) or not prompts:
            return
        prompt = prompts[0]
        self._prompt_path = prompt.path
        self.prompt_text.setText(
            f'"{Path(prompt.path).name}" looks like a dataset ({prompt.image_count:,} images: '
            f"{prompt.signs})."
        )
        self.prompt_text.setToolTip(prompt.path)
        self.prompt_banner.show()

    def _decide_prompt(self, index_it: bool) -> None:
        if self._prompt_path:
            path = self._prompt_path
            self.prompt_banner.hide()
            self._prompt_path = None
            _run(self.pool, lambda: self.engine.decide_dataset(path, index_it), lambda _: self.refresh_status())

    # ---- window -------------------------------------------------------------------------------

    def open_settings(self) -> None:
        if self._settings is None:
            self._settings = SettingsDialog(self.engine, self)
        self._settings.refresh()
        self._settings.show()
        self._settings.raise_()
        self._settings.activateWindow()

    def prepare_search(self) -> None:
        """Load the search model in the background, so the first search doesn't wait for it."""
        _run(self.pool, self.engine.prepare_search)

    def bring_forward(self) -> None:
        self.prepare_search()  # It unloads after a long idle, so opening the window reloads it.
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.search_box.setFocus()
        self.search_box.selectAll()

    def closeEvent(self, event) -> None:
        # Closing hides to the tray, so indexing and the watcher keep running.
        if self.quitting or not QSystemTrayIcon.isSystemTrayAvailable():
            event.accept()
            return
        event.ignore()
        self.hide()


class SettingsDialog(QDialog):
    def __init__(self, engine: Engine, parent: SearchWindow) -> None:
        super().__init__(parent)
        self.engine = engine
        self.window_ = parent
        self.pool = parent.pool
        self.setWindowTitle("LodeStone settings")
        self.resize(720, 500)

        tabs = QTabWidget()
        tabs.addTab(self._folders_tab(), "Folders")
        tabs.addTab(self._review_tab(), "Name-only images")

        self.autostart = QCheckBox("Start LodeStone when I sign in")
        self.autostart.setChecked(_autostart_enabled())
        self.autostart.toggled.connect(_set_autostart)
        self.pause = QCheckBox("Pause indexing")
        self.pause.setChecked(engine.paused)
        self.pause.toggled.connect(engine.set_paused)
        self.use_gpu = QCheckBox("Use GPU if available")
        self.use_gpu.setChecked(engine.use_gpu)
        self.use_gpu.setToolTip(
            "Faster indexing, needs 3-4GB vram"
        )
        self.use_gpu.toggled.connect(
            lambda checked: _run(self.pool, lambda: engine.set_use_gpu(checked))
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)

        footer = QHBoxLayout()
        footer.addWidget(self.autostart)
        footer.addWidget(self.pause)
        footer.addWidget(self.use_gpu)
        footer.addStretch(1)
        footer.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addLayout(footer)

        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh_counts)

    def showEvent(self, event) -> None:
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)

    # ---- folders tab ---------------------------------------------------------------------------

    def _folders_tab(self) -> QWidget:
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(["Indexed folders", "Files", "Status"])
        self.folder_tree.setRootIsDecorated(False)
        self.folder_tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.folder_tree.setColumnWidth(0, 430)
        self.folder_tree.setColumnWidth(1, 70)
        self.folder_tree.itemChanged.connect(self._folder_toggled)

        add = QPushButton("+ Add Folder")
        add.clicked.connect(self._add_folder)
        remove = QPushButton("- Remove")
        remove.clicked.connect(self._remove_folder)
        folder_buttons = QHBoxLayout()
        folder_buttons.addWidget(add)
        folder_buttons.addWidget(remove)
        folder_buttons.addStretch(1)

        self.exclude_list = QListWidget()
        self.exclude_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.exclude_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        add_exclude = QPushButton("+ Exclude Folder")
        add_exclude.clicked.connect(self._add_exclude)
        remove_exclude = QPushButton("- Remove")
        remove_exclude.clicked.connect(self._remove_exclude)
        exclude_buttons = QHBoxLayout()
        exclude_buttons.addWidget(add_exclude)
        exclude_buttons.addWidget(remove_exclude)
        exclude_buttons.addStretch(1)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self.folder_tree, 3)
        layout.addLayout(folder_buttons)
        layout.addWidget(QLabel("Excluded folders"))
        layout.addWidget(self.exclude_list, 2)
        layout.addLayout(exclude_buttons)
        return tab

    def refresh(self) -> None:
        self.folder_tree.blockSignals(True)
        self.folder_tree.clear()
        for folder in self.engine.folders():
            item = QTreeWidgetItem([folder.path, "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, folder.path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if folder.enabled else Qt.CheckState.Unchecked)
            self.folder_tree.addTopLevelItem(item)
        self.folder_tree.blockSignals(False)

        self.exclude_list.clear()
        self.exclude_list.addItems(self.engine.excludes())
        self.pause.setChecked(self.engine.paused)
        self.refresh_counts()
        self.refresh_review()

    def refresh_counts(self) -> None:
        catalog = self.engine.catalog
        paths = [
            self.folder_tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.folder_tree.topLevelItemCount())
        ]
        _run(self.pool, lambda: {p: catalog.folder_counts(p) for p in paths}, self._apply_counts)

    def _apply_counts(self, counts) -> None:
        if isinstance(counts, Exception):
            return
        self.folder_tree.blockSignals(True)
        for i in range(self.folder_tree.topLevelItemCount()):
            item = self.folder_tree.topLevelItem(i)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if path not in counts:
                continue
            indexed, queued = counts[path]
            item.setText(1, f"{indexed:,}")
            if item.checkState(0) != Qt.CheckState.Checked:
                state = "Off"
            elif not os.path.isdir(path):
                state = "Not found"
            elif queued:
                state = f"Indexing, {queued:,} to go"
            else:
                state = "Up to date"
            item.setText(2, state)
        self.folder_tree.blockSignals(False)

    def _folder_toggled(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        enabled = item.checkState(0) == Qt.CheckState.Checked
        _run(self.pool, lambda: self.engine.set_folder_enabled(path, enabled), lambda _: self.refresh_counts())

    def _selected_folder(self) -> str | None:
        item = self.folder_tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder to index")
        if not path:
            return
        path = os.path.normpath(path)
        self.setCursor(Qt.CursorShape.BusyCursor)
        _run(self.pool, lambda: self.engine.estimate(path), lambda estimate: self._confirm_add(path, estimate))

    def _confirm_add(self, path: str, estimate) -> None:
        self.unsetCursor()
        if isinstance(estimate, Exception):
            QMessageBox.warning(self, "Couldn't read that folder", str(estimate))
            return
        if estimate.is_drive_root:
            answer = QMessageBox.warning(
                self,
                "Index a whole drive?",
                f"{path} is a whole drive, with {estimate.total:,} files to index. Most of it is "
                "usually program files and other noise, which makes results worse and indexing "
                "much slower.\n\nIndex the whole drive anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        elif estimate.total >= self.engine.settings.large_folder_warning:
            answer = QMessageBox.question(
                self,
                "Index this large folder?",
                f"{path} has {estimate.images:,} images and {estimate.documents:,} documents, "
                "which could take many hours.\n\nIndex it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        _run(self.pool, lambda: self.engine.add_folder(path), lambda _: self.refresh())

    def _remove_folder(self) -> None:
        path = self._selected_folder()
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "Remove this folder?",
            f"LodeStone will stop indexing {path} and forget what it found there. "
            "Your files won't be touched.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            _run(self.pool, lambda: self.engine.remove_folder(path), lambda _: self.refresh())

    def _add_exclude(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder to exclude")
        if path:
            path = os.path.normpath(path)
            _run(self.pool, lambda: self.engine.add_exclude(path), lambda _: self.refresh())

    def _remove_exclude(self) -> None:
        item = self.exclude_list.currentItem()
        if item:
            path = item.text()
            _run(self.pool, lambda: self.engine.remove_exclude(path), lambda _: self.refresh())

    # ---- review tab ----------------------------------------------------------------------------

    def _review_tab(self) -> QWidget:
        explain = QLabel(
            "These images can only be found by filename, because they failed a quality check. "
            "If one was flagged by mistake, select it and choose Index anyway."
        )
        explain.setWordWrap(True)
        self.review_tree = QTreeWidget()
        self.review_tree.setHeaderLabels(["Image", "Reason", "Folder"])
        self.review_tree.setRootIsDecorated(False)
        self.review_tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.review_tree.setColumnWidth(0, 240)
        self.review_tree.setColumnWidth(1, 90)
        self.review_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        anyway = QPushButton("Index anyway")
        anyway.clicked.connect(self._index_anyway)
        row = QHBoxLayout()
        row.addWidget(anyway)
        row.addStretch(1)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(explain)
        layout.addWidget(self.review_tree, 1)
        layout.addLayout(row)
        return tab

    def refresh_review(self) -> None:
        self.review_tree.clear()
        for row in self.engine.name_only_images():
            reason = SKIP_REASONS.get(row.skip_reason or "", row.skip_reason or "")
            item = QTreeWidgetItem([row.name, reason, str(Path(row.path).parent)])
            item.setData(0, Qt.ItemDataRole.UserRole, row.path)
            self.review_tree.addTopLevelItem(item)

    def _index_anyway(self) -> None:
        for item in self.review_tree.selectedItems():
            self.engine.index_anyway(item.data(0, Qt.ItemDataRole.UserRole))
            index = self.review_tree.indexOfTopLevelItem(item)
            self.review_tree.takeTopLevelItem(index)


def _autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY) as key:
            winreg.QueryValueEx(key, AUTOSTART_NAME)
        return True
    except OSError:
        return False


def _set_autostart(enabled: bool) -> None:
    """Only ever touches LodeStone's own entry, and only when the user ticks the box."""
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            executable = Path(sys.executable).with_name("pythonw.exe")
            winreg.SetValueEx(
                key, AUTOSTART_NAME, 0, winreg.REG_SZ,
                f'"{executable}" -m lodestone.ui.app --background',
            )
        else:
            try:
                winreg.DeleteValue(key, AUTOSTART_NAME)
            except FileNotFoundError:
                pass


def _instance_key() -> str:
    """One name per index folder, so only one copy of the app can ever write to an index."""
    digest = hashlib.sha1(os.path.normcase(str(data_dir())).encode()).hexdigest()[:16]
    return f"lodestone-{digest}"


def _ask_running_copy_to_show() -> bool:
    socket = QLocalSocket()
    socket.connectToServer(_instance_key())
    if not socket.waitForConnected(500):
        return False
    socket.write(b"show")
    socket.waitForBytesWritten(500)
    socket.disconnectFromServer()
    return True


def _listen_for_other_copies(window: "SearchWindow", app: QApplication) -> QLocalServer:
    server = QLocalServer(app)
    # A crash can leave the old name behind, and nothing else is using it at this point.
    QLocalServer.removeServer(_instance_key())
    server.listen(_instance_key())
    server.newConnection.connect(window.bring_forward)
    return server


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    app = QApplication(sys.argv)
    app.setApplicationName("LodeStone")
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())

    if _ask_running_copy_to_show():
        return 0

    engine = Engine()
    engine.seed_defaults()
    engine.start()
    window = SearchWindow(engine)
    _server = _listen_for_other_copies(window, app)

    def quit_app() -> None:
        window.quitting = True
        window.close()
        app.quit()

    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(app_icon(), app)
        tray.setToolTip("LodeStone")
        menu = QMenu()
        open_action = QAction("Open LodeStone", menu)
        open_action.triggered.connect(window.bring_forward)
        settings_action = QAction("Settings", menu)
        settings_action.triggered.connect(window.open_settings)
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(quit_app)
        menu.addAction(open_action)
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: window.bring_forward()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        tray.show()
    else:
        app.setQuitOnLastWindowClosed(True)

    # Starting in the background at sign-in shouldn't load a model nobody has asked for yet.
    if "--background" not in sys.argv:
        window.show()
        window.prepare_search()

    code = app.exec()
    window.pool.waitForDone(3000)
    engine.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
