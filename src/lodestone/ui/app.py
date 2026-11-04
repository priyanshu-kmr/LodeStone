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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSystemTrayIcon,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_MODEL,
    MEDIA_SECONDS_MAX,
    MEDIA_SECONDS_STEP,
    SETTINGS,
    TOKEN_LIMIT_MAX,
    TOKEN_LIMIT_STEP,
    EmbeddingModel,
    data_dir,
)
from ..engine import Engine
from ..scanner import parse_exts
from ..search import ALL, AUDIO, DOCUMENTS, IMAGES, VIDEOS, SearchResult

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


def _duration_text(seconds: int) -> str:
    minutes, rest = divmod(seconds, 60)
    if not minutes:
        return f"{rest}s"
    return f"{minutes}m {rest}s" if rest else f"{minutes}m"


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
        for label, mode in (
            ("All", ALL), ("Documents", DOCUMENTS), ("Images", IMAGES), ("Videos", VIDEOS), ("Audio", AUDIO),
        ):
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

        self.more_button = QPushButton("Show more results")
        self.more_button.clicked.connect(self._show_more)
        self.more_button.hide()
        self._shown_search: tuple[str, str] = ("", ALL)

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
        layout.addWidget(self.more_button)
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
        top_k = self.engine.settings.default_top_k
        self._shown_search = (query, mode)
        self.more_button.setEnabled(False)  # It would add to the results about to be replaced.
        if not self.engine.search_model_loaded:
            self._waiting_for_model = True
            self.status_text.setText("Loading the search model, which takes a few seconds...")
        _run(
            self.pool,
            lambda: self.engine.search(query, top_k, mode),
            lambda results: self._show_results(request, query, top_k, results),
        )

    def _show_results(self, request: int, query: str, top_k: int, results) -> None:
        if request != self._request:
            return  # A newer search has started, so this answer is out of date.
        self._waiting_for_model = False
        self.results.clear()
        self.more_button.setEnabled(True)
        if isinstance(results, Exception):
            self._show_empty("Search failed", str(results))
            return
        if not results:
            self._show_empty(f'Nothing found for "{query}"', "Try describing it differently, or check which folders are indexed in Settings.")
            return
        self.empty.hide()
        self.results.show()
        self._add_results(request, results)
        self.results.setCurrentRow(0)
        # A full page means there may be more. Fewer means that was everything.
        self.more_button.setVisible(len(results) >= top_k)

    def _add_results(self, request: int, results) -> None:
        for result in results:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, result.path)
            row = ResultRow(result)
            item.setSizeHint(QSize(0, THUMB + 14))
            self.results.addItem(item)
            self.results.setItemWidget(item, row)
            if result.kind == "image":
                self._load_thumbnail(request, result.path, row)

    def _show_more(self) -> None:
        request = self._request  # The same search, so thumbnails still loading aren't thrown away.
        query, mode = self._shown_search
        top_k = self.results.count() + self.engine.settings.default_top_k
        self.more_button.setEnabled(False)
        _run(
            self.pool,
            lambda: self.engine.search(query, top_k, mode),
            lambda results: self._add_more_results(request, top_k, results),
        )

    def _add_more_results(self, request: int, top_k: int, results) -> None:
        if request != self._request:
            return
        self.more_button.setEnabled(True)
        if isinstance(results, Exception):
            self.status_text.setText(f"Couldn't load more results: {results}")
            return
        # Searching deeper can reorder results slightly. Only add to the end, so nothing moves
        # while it's being read.
        shown = {self.results.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.results.count())}
        new = [result for result in results if result.path not in shown]
        first_new = self.results.count()
        self._add_results(request, new)
        if new:
            self.results.setCurrentRow(first_new)
        self.more_button.setVisible(bool(new) and len(results) >= top_k)

    def _load_thumbnail(self, request: int, path: str, row: ResultRow) -> None:
        def show(image) -> None:
            if request == self._request and image is not None and not isinstance(image, Exception):
                row.set_thumbnail(image)

        _run(self.pool, lambda: load_thumbnail(path), show)

    def _show_empty(self, title: str, body: str) -> None:
        self.results.hide()
        self.more_button.hide()
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
        tabs.addTab(self._index_tab(), "Index")
        tabs.addTab(self._advanced_tab(), "Advanced")

        self.autostart = QCheckBox("Start LodeStone when I sign in")
        self.autostart.setChecked(_autostart_enabled())
        self.autostart.toggled.connect(_set_autostart)
        self.pause = QCheckBox("Pause indexing")
        self.pause.setChecked(engine.paused)
        self.pause.toggled.connect(lambda checked: _run(self.pool, lambda: engine.set_paused(checked)))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)

        footer = QHBoxLayout()
        footer.addWidget(self.autostart)
        footer.addWidget(self.pause)
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

    def keyPressEvent(self, event) -> None:
        # Enter in the file type box would otherwise also press the dialog's default button.
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)

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
        self.type_list.clear()
        self.type_list.addItems(self.engine.custom_exts())
        self.token_slider.setValue(self.engine.token_limit // TOKEN_LIMIT_STEP)
        self._token_moved()
        self.video_slider.setValue(self.engine.video_seconds // MEDIA_SECONDS_STEP)
        self.audio_slider.setValue(self.engine.audio_seconds // MEDIA_SECONDS_STEP)
        self._media_moved()
        if self.model_input.isEnabled():  # Otherwise a model is being tested, which can take minutes.
            self._show_model()
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
                state = f"{'Paused' if self.engine.paused else 'Indexing'}, {queued:,} to go"
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
                f"{path} has {estimate.images:,} images, {estimate.documents:,} documents and "
                f"{estimate.media:,} audio and video files, which could take many hours.\n\nIndex it anyway?",
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
            "If one was flagged by mistake, select it and choose Index."
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

    # ---- index tab -----------------------------------------------------------------------------

    def _index_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._types_group(), 1)
        layout.addWidget(self._token_group())
        layout.addWidget(self._media_group())
        layout.addWidget(self._manage_group())
        return tab

    def _types_group(self) -> QGroupBox:
        explain = QLabel(
            "Search inside more file types. They need to be plain text, like .xml or .py, or images "
            "that OCR can read, like .jfif. Anything else, like .zip or .exe, can only be found by "
            "its filename."
        )
        explain.setWordWrap(True)
        self.type_list = QListWidget()
        self.type_input = QLineEdit()
        self.type_input.setPlaceholderText(".xml, .py")
        self.type_input.setAccessibleName("File types to add")
        self.type_input.returnPressed.connect(self._add_types)
        add = QPushButton("+ Add")
        add.clicked.connect(self._add_types)
        remove = QPushButton("- Remove")
        remove.clicked.connect(self._remove_type)
        row = QHBoxLayout()
        row.addWidget(self.type_input, 1)
        row.addWidget(add)
        row.addWidget(remove)

        group = QGroupBox("Extra file types")
        layout = QVBoxLayout(group)
        layout.addWidget(explain)
        layout.addWidget(self.type_list, 1)
        layout.addLayout(row)
        return group

    def _token_group(self) -> QGroupBox:
        self.token_slider = QSlider(Qt.Orientation.Horizontal)
        self.token_slider.setRange(0, TOKEN_LIMIT_MAX // TOKEN_LIMIT_STEP)
        self.token_slider.setPageStep(5)
        self.token_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.token_slider.setTickInterval(5)
        self.token_slider.setAccessibleName("Token limit")
        self.token_slider.valueChanged.connect(self._token_moved)
        self.token_value = QLabel()
        self.token_value.setMinimumWidth(80)
        self.token_apply = QPushButton("Apply")
        self.token_apply.clicked.connect(self._apply_token_limit)
        row = QHBoxLayout()
        row.addWidget(self.token_slider, 1)
        row.addWidget(self.token_value)
        row.addWidget(self.token_apply)
        hint = QLabel(
            "Tokens embedded per file for semantic search. Default: "
            f"{SETTINGS.embed_token_limit}. Higher values improve long-file retrieval but slow indexing "
            "and increase index size. At 0, only exact words and filenames are searchable; scanned PDFs "
            "are not read."
        )
        hint.setWordWrap(True)
        hint.setObjectName("statusText")

        group = QGroupBox("Token limit")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        layout.addWidget(hint)
        return group

    def _media_row(self, name: str, apply: Callable[[], None]) -> tuple[QSlider, QLabel, QPushButton, QHBoxLayout]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, MEDIA_SECONDS_MAX // MEDIA_SECONDS_STEP)
        slider.setPageStep(4)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(4)
        slider.setAccessibleName(f"{name} duration")
        slider.valueChanged.connect(self._media_moved)
        value = QLabel()
        value.setMinimumWidth(60)
        button = QPushButton("Apply")
        button.clicked.connect(apply)
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        row.addWidget(slider, 1)
        row.addWidget(value)
        row.addWidget(button)
        return slider, value, button, row

    def _media_group(self) -> QGroupBox:
        self.video_slider, self.video_value, self.video_apply, video_row = self._media_row(
            "Video", self._apply_video_duration
        )
        self.audio_slider, self.audio_value, self.audio_apply, audio_row = self._media_row(
            "Audio", self._apply_audio_duration
        )
        hint = QLabel(
            "How much of each file to read for semantic search, from the start. Longer durations find "
            "more but slow indexing."
        )
        hint.setWordWrap(True)
        hint.setObjectName("statusText")

        group = QGroupBox("Video and audio duration")
        layout = QVBoxLayout(group)
        layout.addLayout(video_row)
        layout.addLayout(audio_row)
        layout.addWidget(hint)
        return group

    def _manage_group(self) -> QGroupBox:
        self.reindex_button = QPushButton("Reindex")
        self.reindex_button.clicked.connect(self._reindex)
        self.delete_button = QPushButton("Delete index")
        self.delete_button.clicked.connect(self._delete_index)
        self.index_note = QLabel()
        self.index_note.setObjectName("statusText")
        self.index_note.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.reindex_button)
        row.addWidget(self.delete_button)
        row.addWidget(self.index_note, 1)

        group = QGroupBox("Whole index")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        return group

    def _add_types(self) -> None:
        try:
            exts = parse_exts(self.type_input.text())
        except ValueError as error:
            QMessageBox.warning(self, "Couldn't add that file type", str(error))
            return
        if exts:
            self.type_input.clear()
            _run(self.pool, lambda: self.engine.add_custom_exts(exts), lambda _: self.refresh())

    def _remove_type(self) -> None:
        item = self.type_list.currentItem()
        if item:
            ext = item.text()
            _run(self.pool, lambda: self.engine.remove_custom_ext(ext), lambda _: self.refresh())

    def _token_moved(self) -> None:
        limit = self.token_slider.value() * TOKEN_LIMIT_STEP
        self.token_value.setText(f"{limit:,} tokens")
        self.token_apply.setEnabled(limit != self.engine.token_limit)

    def _paused_note(self) -> str:
        return " Indexing is paused, so this starts when you turn it back on." if self.engine.paused else ""

    def _apply_token_limit(self) -> None:
        limit = self.token_slider.value() * TOKEN_LIMIT_STEP
        answer = QMessageBox.question(
            self,
            "Reindex with the new limit?",
            f"LodeStone will read and embed your documents again using {limit:,} tokens, which could "
            "take a while. Images are only redone if they have text in them. Search keeps working "
            f"in the meantime.{self._paused_note()}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.token_slider.setValue(self.engine.token_limit // TOKEN_LIMIT_STEP)
            return
        self._index_job(lambda: self.engine.set_token_limit(limit), f"Reindexing with {limit:,} tokens.")

    def _media_moved(self) -> None:
        video = self.video_slider.value() * MEDIA_SECONDS_STEP
        audio = self.audio_slider.value() * MEDIA_SECONDS_STEP
        self.video_value.setText(_duration_text(video))
        self.audio_value.setText(_duration_text(audio))
        self.video_apply.setEnabled(video != self.engine.video_seconds)
        self.audio_apply.setEnabled(audio != self.engine.audio_seconds)

    def _confirm_media_reindex(self, what: str, seconds: int) -> bool:
        answer = QMessageBox.question(
            self,
            "Reindex with the new duration?",
            f"LodeStone will read and embed your {what} again using the first {_duration_text(seconds)}, "
            f"which could take a while. Search keeps working in the meantime.{self._paused_note()}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _apply_video_duration(self) -> None:
        seconds = self.video_slider.value() * MEDIA_SECONDS_STEP
        if self._confirm_media_reindex("videos", seconds):
            self._index_job(
                lambda: self.engine.set_video_seconds(seconds),
                f"Reindexing videos with the first {_duration_text(seconds)}.",
            )
        else:
            self.video_slider.setValue(self.engine.video_seconds // MEDIA_SECONDS_STEP)

    def _apply_audio_duration(self) -> None:
        seconds = self.audio_slider.value() * MEDIA_SECONDS_STEP
        if self._confirm_media_reindex("audio and videos", seconds):
            self._index_job(
                lambda: self.engine.set_audio_seconds(seconds),
                f"Reindexing audio with the first {_duration_text(seconds)}.",
            )
        else:
            self.audio_slider.setValue(self.engine.audio_seconds // MEDIA_SECONDS_STEP)

    def _reindex(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reindex everything?",
            "LodeStone will read and embed all your files again, which could take many hours. "
            f"Search keeps working in the meantime.{self._paused_note()}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._index_job(self.engine.reindex, "Reindexing.")

    def _delete_index(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Delete the whole index?",
            "LodeStone will forget everything it has indexed and free up the disk space. Your files "
            "won't be touched, and your folders and settings stay.\n\nSearch won't find anything "
            "until it's rebuilt. Indexing gets paused, so turn off Pause indexing when you want that.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._index_job(self.engine.delete_index, "Index deleted. Indexing is paused.")

    def _index_job(self, work: Callable[[], None], message: str) -> None:
        """These wait for the current indexing batch to finish, so they can take a moment."""
        self.reindex_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.token_apply.setEnabled(False)
        self.video_apply.setEnabled(False)
        self.audio_apply.setEnabled(False)
        self.index_note.setText("Working on it...")
        _run(self.pool, work, lambda result: self._index_job_done(result, message))

    def _index_job_done(self, result: object, message: str) -> None:
        self.reindex_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        if isinstance(result, Exception):
            self.index_note.setText("")
            QMessageBox.warning(self, "That didn't work", str(result))
        else:
            self.index_note.setText(message)
        self.refresh()

    # ---- advanced tab --------------------------------------------------------------------------

    def _advanced_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._model_group())
        layout.addWidget(self._gpu_group())
        layout.addWidget(self._unload_group())
        layout.addStretch(1)
        return tab

    def _option(self, text: str, checked: bool, apply: Callable[[bool], None]) -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(checked)
        box.toggled.connect(lambda value: _run(self.pool, lambda: apply(value)))
        return box

    def _options_group(self, title: str, boxes: list[QCheckBox], explanation: str) -> QGroupBox:
        hint = QLabel(explanation)
        hint.setWordWrap(True)
        hint.setObjectName("statusText")
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        for box in boxes:
            layout.addWidget(box)
        layout.addWidget(hint)
        return group

    def _gpu_group(self) -> QGroupBox:
        engine = self.engine
        self.gpu_indexing = self._option(
            "Use GPU for the indexing model", engine.gpu_for_indexing, engine.set_gpu_for_indexing
        )
        self.gpu_search = self._option(
            "Use GPU for the search model", engine.gpu_for_search, engine.set_gpu_for_search
        )
        return self._options_group(
            "GPU",
            [self.gpu_indexing, self.gpu_search],
            "Indexing becomes significantly faster but needs upto 4 GB of VRAM"
            )

    def _unload_group(self) -> QGroupBox:
        engine = self.engine
        self.unload_index = self._option(
            f"Unload the indexing model after {_idle_text(SETTINGS.vision_idle_unload_seconds)} idle",
            engine.settings.unload_index_model,
            engine.set_unload_index_model,
        )
        self.unload_search = self._option(
            f"Unload the search model after {_idle_text(SETTINGS.search_idle_unload_seconds)} idle",
            engine.settings.unload_search_model,
            engine.set_unload_search_model,
        )
        return self._options_group(
            "Automatic unloading",
            [self.unload_index, self.unload_search],
            "A model that stays loaded responds right away, but keeps using RAM, or VRAM on the GPU, "
            "until LodeStone quits. Search can take a few seconds to load the model again.",
        )

    def _model_group(self) -> QGroupBox:
        self.model_current = QLabel()
        self.model_current.setWordWrap(True)
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("sentence-transformers/all-MiniLM-L6-v2")
        self.model_input.setAccessibleName("Hugging Face model ID")
        self.model_input.textChanged.connect(self._model_edited)
        self.model_input.returnPressed.connect(self._switch_to_typed_model)
        self.model_switch = QPushButton("Switch")
        self.model_switch.clicked.connect(self._switch_to_typed_model)
        self.model_default = QPushButton("Use default")
        self.model_default.clicked.connect(lambda: self._confirm_model(DEFAULT_MODEL))
        row = QHBoxLayout()
        row.addWidget(self.model_input, 1)
        row.addWidget(self.model_switch)
        row.addWidget(self.model_default)
        hint = QLabel(
            "Enter a Hugging Face sentence-transformers model. It's tested before switching, then "
            "all files are reindexed. Unsupported file types are found by filename and text only."
        )
        hint.setWordWrap(True)
        hint.setObjectName("statusText")
        self.model_note = QLabel()
        self.model_note.setObjectName("statusText")
        self.model_note.setWordWrap(True)

        group = QGroupBox("Embedding model")
        layout = QVBoxLayout(group)
        layout.addWidget(self.model_current)
        layout.addLayout(row)
        layout.addWidget(hint)
        layout.addWidget(self.model_note)
        return group

    def _show_model(self) -> None:
        model = self.engine.embedding_model
        self.model_current.setText(f"Current model: {_describe_model(model)}")
        self.model_input.setText(model.id)
        self.model_default.setEnabled(model.id != DEFAULT_MODEL.id)
        self._model_edited()

    def _model_edited(self) -> None:
        typed = self.model_input.text().strip()
        self.model_switch.setEnabled(bool(typed) and typed != self.engine.embedding_model.id)

    def _switch_to_typed_model(self) -> None:
        typed = self.model_input.text().strip()
        if not typed or typed == self.engine.embedding_model.id:
            return
        if typed == DEFAULT_MODEL.id:
            self._confirm_model(DEFAULT_MODEL)
            return
        self._set_model_busy(f"Downloading and testing {typed}, which can take several minutes...")
        _run(self.pool, lambda: self.engine.inspect_model(typed), self._confirm_model)

    def _set_model_busy(self, message: str) -> None:
        self.model_input.setEnabled(False)
        self.model_switch.setEnabled(False)
        self.model_default.setEnabled(False)
        self.model_note.setText(message)

    def _model_job_done(self, message: str) -> None:
        self.model_input.setEnabled(True)
        self.model_note.setText(message)
        self._show_model()

    def _confirm_model(self, result: EmbeddingModel | Exception) -> None:
        if isinstance(result, Exception):
            self._model_job_done("")
            QMessageBox.warning(self, "Couldn't use that model", str(result))
            return
        answer = QMessageBox.question(
            self,
            "Switch model and reindex?",
            f"Switch to {_describe_model(result)}\n\nLodeStone will delete the current embeddings and "
            "embed all your files again, which could take many hours. Until a file's turn comes, it "
            f"can only be found by its filename and exact words.{self._paused_note()}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._model_job_done("")
            return
        self._set_model_busy("Working on it...")
        _run(self.pool, lambda: self.engine.set_embedding_model(result), lambda done: self._model_switched(result, done))

    def _model_switched(self, model: EmbeddingModel, result: object) -> None:
        if isinstance(result, Exception):
            self._model_job_done("")
            QMessageBox.warning(self, "That didn't work", str(result))
        else:
            self._model_job_done(f"Reindexing with {model.id}.")


def _idle_text(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def _describe_model(model: EmbeddingModel) -> str:
    names = {"text": "text", "image": "images", "audio": "audio", "video": "video"}
    kinds = [names[kind] for kind in names if kind in model.kinds]
    read = ", ".join(kinds[:-1]) + f" and {kinds[-1]}" if len(kinds) > 1 else kinds[0]
    return f"{model.id}, which reads {read} ({model.dim} dimensions)."


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
