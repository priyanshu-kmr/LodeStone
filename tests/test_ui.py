"""Smoke tests for the UI, run without showing a real window."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QObject  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget  # noqa: E402

from conftest import FakeEmbedder, FakeOcr, textured  # noqa: E402
from lodestone.config import Settings  # noqa: E402
from lodestone.engine import Engine  # noqa: E402
from lodestone.ui.app import STYLE, SearchWindow, SettingsDialog  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    return app


@pytest.fixture
def engine(paths, library):
    eng = Engine(paths, FakeEmbedder(), FakeOcr(), Settings(debounce_seconds=0, delete_delay_seconds=0), watch=False)
    eng.catalog.add_folder(str(library))
    yield eng
    eng.stop()


def drain(engine: Engine) -> None:
    engine.scan_all()
    while engine.indexer.run_batch():
        pass
    engine.indexer.maintain()


def wait_for(app, check, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if check():
            return True
        time.sleep(0.01)
    return False


def finish(window: SearchWindow, app) -> None:
    window.quitting = True
    window.status_timer.stop()
    window.close()
    window.pool.waitForDone(3000)
    app.processEvents()


def test_typing_shows_results_with_snippets_and_thumbnails(qapp, engine, library) -> None:
    (library / "interview_notes.txt").write_text("We discussed system design and ownership.")
    textured((640, 480), (220, 20, 20)).save(library / "red_photo.png")
    drain(engine)

    window = SearchWindow(engine)
    window.show()
    window.search_box.setText("system design")
    assert wait_for(qapp, lambda: window.results.count() > 0)
    assert window.results.item(0).data(256).endswith("interview_notes.txt")
    assert window.results.isVisible() and not window.empty.isVisible()

    window.search_box.setText("")
    assert window.empty.isVisible() and window.results.count() == 0

    window.search_box.setText("photo of red")
    assert wait_for(qapp, lambda: window.results.count() == 1)
    row = window.results.itemWidget(window.results.item(0))
    assert wait_for(qapp, lambda: row.thumb.pixmap() is not None and not row.thumb.pixmap().isNull())
    finish(window, qapp)


class _WindowSpy(QObject):
    """Records every top-level window that gets shown, even if only for a moment."""

    def __init__(self) -> None:
        super().__init__()
        self.windows: list[str] = []

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            self.windows.append(type(watched).__name__)
        return False


def test_searching_never_opens_extra_windows(qapp, engine, library) -> None:
    (library / "notes.txt").write_text("We discussed system design and ownership at length.")
    textured((640, 480), (220, 20, 20)).save(library / "red_photo.png")
    drain(engine)

    window = SearchWindow(engine)
    window.show()
    wait_for(qapp, lambda: window.isVisible())
    spy = _WindowSpy()
    qapp.installEventFilter(spy)
    try:
        window.search_box.setText("system design")
        assert wait_for(qapp, lambda: window.results.count() > 0)
        wait_for(qapp, lambda: False, timeout=0.3)
    finally:
        qapp.removeEventFilter(spy)
    assert spy.windows == []
    finish(window, qapp)


def test_no_results_explains_what_to_do(qapp, engine) -> None:
    window = SearchWindow(engine)
    window.show()
    window.search_box.setText("nothing is indexed yet")
    assert wait_for(qapp, lambda: "Nothing found" in window.empty_title.text())
    finish(window, qapp)


def test_filter_buttons_change_the_results(qapp, engine, library) -> None:
    (library / "red.txt").write_text("red red red")
    textured((640, 480), (220, 20, 20)).save(library / "red.png")
    drain(engine)
    window = SearchWindow(engine)
    window.show()
    window.search_box.setText("red")
    assert wait_for(qapp, lambda: window.results.count() == 2)
    images = next(b for b in window.filters.buttons() if b.text() == "Images")
    images.click()
    assert wait_for(qapp, lambda: window.results.count() == 1)
    assert window.results.item(0).data(256).endswith("red.png")
    finish(window, qapp)


def test_status_line_and_dataset_prompt(qapp, engine, library) -> None:
    (library / "a.txt").write_text("hello")
    drain(engine)
    engine.catalog.record_dataset_prompt(str(library / "data"), 75830, "numbered filenames, code nearby")
    window = SearchWindow(engine)
    window.show()
    assert wait_for(qapp, lambda: "1 files indexed" in window.status_text.text())
    assert wait_for(qapp, lambda: window.prompt_banner.isVisible())
    assert "75,830" in window.prompt_text.text()
    window._decide_prompt(False)
    assert wait_for(qapp, lambda: str(library / "data") in engine.excludes())
    finish(window, qapp)


def test_settings_lists_and_changes_folders(qapp, engine, library, tmp_path, monkeypatch) -> None:
    (library / "a.txt").write_text("hello")
    drain(engine)
    window = SearchWindow(engine)
    dialog = SettingsDialog(engine, window)
    dialog.refresh()
    dialog.show()
    assert dialog.folder_tree.topLevelItemCount() == 1
    assert wait_for(qapp, lambda: dialog.folder_tree.topLevelItem(0).text(1) == "1")

    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(extra))
    dialog._add_folder()
    assert wait_for(qapp, lambda: dialog.folder_tree.topLevelItemCount() == 2)

    dialog._add_exclude()
    assert wait_for(qapp, lambda: dialog.exclude_list.count() == 1)
    dialog.exclude_list.setCurrentRow(0)
    dialog._remove_exclude()
    assert wait_for(qapp, lambda: dialog.exclude_list.count() == 0)

    dialog.folder_tree.setCurrentItem(dialog.folder_tree.topLevelItem(1))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog._remove_folder()
    assert wait_for(qapp, lambda: dialog.folder_tree.topLevelItemCount() == 1)
    dialog.close()
    finish(window, qapp)


def test_large_folder_needs_confirmation(qapp, engine, tmp_path, monkeypatch) -> None:
    big = tmp_path / "big"
    big.mkdir()
    for i in range(6):
        (big / f"{i}.txt").write_text("x")
    settings = Settings(large_folder_warning=5)
    object.__setattr__(engine, "settings", settings)
    window = SearchWindow(engine)
    dialog = SettingsDialog(engine, window)
    dialog.refresh()
    seen = []

    def decline(*args, **kwargs):
        seen.append(args[2])
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(big))
    monkeypatch.setattr(QMessageBox, "question", decline)
    dialog._add_folder()
    assert wait_for(qapp, lambda: seen)
    assert "6 documents" in seen[0]
    window.pool.waitForDone(2000)
    assert str(big) not in [f.path for f in engine.folders()]
    dialog.close()
    finish(window, qapp)


def test_review_tab_can_promote_an_image(qapp, engine, library) -> None:
    textured((100, 100), (10, 200, 10)).save(library / "tiny.png")
    drain(engine)
    window = SearchWindow(engine)
    dialog = SettingsDialog(engine, window)
    dialog.refresh()
    assert dialog.review_tree.topLevelItemCount() == 1
    assert dialog.review_tree.topLevelItem(0).text(1) == "Too small"
    dialog.review_tree.topLevelItem(0).setSelected(True)
    dialog._index_anyway()
    assert dialog.review_tree.topLevelItemCount() == 0
    while engine.indexer.run_batch():
        pass
    assert engine.catalog.get_file(str(library / "tiny.png")).index_level == "full"
    dialog.close()
    finish(window, qapp)
