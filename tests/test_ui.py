"""Smoke tests for the UI, run without showing a real window."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QObject, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox, QWidget  # noqa: E402

from conftest import FakeEmbedder, FakeOcr, textured  # noqa: E402
from lodestone.config import MEDIA_SECONDS_STEP, TOKEN_LIMIT_STEP, Settings  # noqa: E402
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


def test_gpu_checkbox_starts_off_and_saves_the_choice(qapp, engine) -> None:
    window = SearchWindow(engine)
    dialog = SettingsDialog(engine, window)
    assert dialog.use_gpu.isChecked() is False
    dialog.use_gpu.setChecked(True)
    assert wait_for(qapp, lambda: engine.use_gpu)
    assert engine.embedder.gpu_settings == [True]
    dialog.close()
    finish(window, qapp)


def _yes(*args, **kwargs):
    return QMessageBox.StandardButton.Yes


def _cancel(*args, **kwargs):
    return QMessageBox.StandardButton.Cancel


@pytest.fixture
def index_tab(qapp, engine, library):
    (library / "notes.txt").write_text("We discussed system design and ownership.")
    drain(engine)
    window = SearchWindow(engine)
    dialog = SettingsDialog(engine, window)
    dialog.refresh()
    yield dialog
    dialog.close()
    finish(window, qapp)


def test_file_types_can_be_added_and_removed(qapp, engine, library, index_tab) -> None:
    dialog = index_tab
    (library / "tool.py").write_text("def quarterly_budget(): pass")
    dialog.type_input.setText("*.PY, xml")
    dialog._add_types()
    assert wait_for(qapp, lambda: dialog.type_list.count() == 2)
    assert dialog.type_input.text() == ""
    assert engine.custom_exts() == [".py", ".xml"]
    drain(engine)
    assert engine.catalog.get_file(str(library / "tool.py")).status == "indexed"

    dialog.type_list.setCurrentRow(0)
    dialog._remove_type()
    assert wait_for(qapp, lambda: dialog.type_list.count() == 1)
    drain(engine)
    assert engine.catalog.get_file(str(library / "tool.py")) is None


def test_file_type_message_mentions_text_and_ocr(index_tab) -> None:
    texts = [label.text() for label in index_tab.findChildren(QLabel)]
    message = next(text for text in texts if "Search inside more file types" in text)
    assert "plain text" in message and "OCR" in message


def test_bad_file_type_is_explained_and_nothing_is_saved(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a[2]))
    dialog.type_input.setText(".py, .pdf")
    dialog._add_types()
    assert "already handles .pdf" in seen[0]
    assert dialog.type_input.text() == ".py, .pdf"
    dialog.window_.pool.waitForDone(2000)
    assert engine.custom_exts() == []


def test_enter_in_the_file_type_box_only_adds_the_type(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: opened.append(True) or "")
    dialog.show()
    dialog.type_input.setText(".py")
    QTest.keyClick(dialog.type_input, Qt.Key.Key_Return)
    assert wait_for(qapp, lambda: dialog.type_list.count() == 1)
    assert dialog.isVisible()
    assert opened == []


def test_slider_starts_at_the_limit_and_covers_the_range(index_tab) -> None:
    dialog = index_tab
    assert dialog.token_slider.value() * TOKEN_LIMIT_STEP == 500
    assert "500 tokens" in dialog.token_value.text()
    assert not dialog.token_apply.isEnabled()
    assert dialog.token_slider.minimum() == 0
    assert dialog.token_slider.maximum() * TOKEN_LIMIT_STEP == 5000


def test_changing_the_token_limit_prompts_then_reindexes(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    seen = []

    def accept(*args, **kwargs):
        seen.append(args[2])
        return QMessageBox.StandardButton.Yes

    calls = engine.embedder.document_calls
    dialog.token_slider.setValue(30)
    assert "3,000 tokens" in dialog.token_value.text()
    assert dialog.token_apply.isEnabled()
    assert engine.token_limit == 500  # Nothing changes until it's confirmed.

    monkeypatch.setattr(QMessageBox, "question", accept)
    dialog.token_apply.click()
    assert "3,000 tokens" in seen[0]
    assert wait_for(qapp, lambda: "Reindexing" in dialog.index_note.text())
    assert engine.token_limit == 3000
    assert not dialog.token_apply.isEnabled()
    drain(engine)
    assert engine.embedder.document_calls == calls + 1


def test_cancelling_the_prompt_puts_the_slider_back(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    calls = engine.embedder.document_calls
    monkeypatch.setattr(QMessageBox, "question", _cancel)
    dialog.token_slider.setValue(30)
    dialog.token_apply.click()
    assert dialog.token_slider.value() * TOKEN_LIMIT_STEP == 500
    assert engine.token_limit == 500
    assert not dialog.token_apply.isEnabled()
    drain(engine)
    assert engine.embedder.document_calls == calls


def test_media_sliders_start_at_the_durations_and_go_from_0s_to_5_minutes(index_tab, engine) -> None:
    dialog = index_tab
    assert dialog.video_slider.value() * MEDIA_SECONDS_STEP == engine.video_seconds
    assert dialog.audio_slider.value() * MEDIA_SECONDS_STEP == engine.audio_seconds
    assert not dialog.video_apply.isEnabled() and not dialog.audio_apply.isEnabled()
    for slider in (dialog.video_slider, dialog.audio_slider):
        assert slider.minimum() == 0
        assert slider.maximum() * MEDIA_SECONDS_STEP == 300
    dialog.video_slider.setValue(20)
    assert dialog.video_value.text() == "5m"
    dialog.video_slider.setValue(6)
    assert dialog.video_value.text() == "1m 30s"
    dialog.audio_slider.setValue(3)
    assert dialog.audio_value.text() == "45s"
    dialog.audio_slider.setValue(0)
    assert dialog.audio_value.text() == "0s"
    assert "token" not in dialog.video_value.text().lower()


def test_changing_a_media_duration_prompts_then_applies(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    start = engine.video_seconds
    new = 4 if start != 4 * MEDIA_SECONDS_STEP else 8
    dialog.video_slider.setValue(new)
    assert dialog.video_apply.isEnabled()
    assert engine.video_seconds == start  # Nothing changes until it's confirmed.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.video_apply.click()
    assert wait_for(qapp, lambda: "Reindexing videos" in dialog.index_note.text())
    assert engine.video_seconds == new * MEDIA_SECONDS_STEP
    assert not dialog.video_apply.isEnabled()


def test_cancelling_a_media_duration_puts_the_slider_back(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    start = engine.audio_seconds
    monkeypatch.setattr(QMessageBox, "question", _cancel)
    dialog.audio_slider.setValue(1)
    dialog.audio_apply.click()
    assert dialog.audio_slider.value() * MEDIA_SECONDS_STEP == start
    assert engine.audio_seconds == start
    assert not dialog.audio_apply.isEnabled()


def test_reindex_button_asks_first(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    calls = engine.embedder.document_calls
    monkeypatch.setattr(QMessageBox, "question", _cancel)
    dialog.reindex_button.click()
    drain(engine)
    assert engine.embedder.document_calls == calls

    monkeypatch.setattr(QMessageBox, "question", _yes)
    dialog.reindex_button.click()
    assert wait_for(qapp, lambda: "Reindexing" in dialog.index_note.text())
    assert dialog.reindex_button.isEnabled()
    drain(engine)
    assert engine.embedder.document_calls == calls + 1


def test_delete_button_asks_first_then_pauses(qapp, engine, index_tab, monkeypatch) -> None:
    dialog = index_tab
    monkeypatch.setattr(QMessageBox, "warning", _cancel)
    dialog.delete_button.click()
    assert wait_for(qapp, lambda: True, timeout=0.1)
    dialog.window_.pool.waitForDone(2000)
    assert engine.status().indexed == 1 and not engine.paused

    monkeypatch.setattr(QMessageBox, "warning", _yes)
    dialog.delete_button.click()
    assert wait_for(qapp, lambda: "Index deleted" in dialog.index_note.text())
    assert engine.status().indexed == 0
    assert engine.paused and dialog.pause.isChecked()
    assert dialog.delete_button.isEnabled()

    # The rescan queues everything again, but nothing runs yet and the folder list says so.
    engine.scan_all()
    dialog.refresh_counts()
    assert wait_for(qapp, lambda: dialog.folder_tree.topLevelItem(0).text(2) == "Paused, 1 to go")

    dialog.pause.setChecked(False)
    assert wait_for(qapp, lambda: not engine.paused)


def test_status_says_when_the_search_model_is_loading(qapp, engine, library) -> None:
    (library / "a.txt").write_text("some notes about a project")
    drain(engine)
    engine.embedder.search_model_loaded = False
    window = SearchWindow(engine)
    window.show()
    release = threading.Event()
    original = engine.search

    def slow_search(*args, **kwargs):
        release.wait(5)
        return original(*args, **kwargs)

    engine.search = slow_search
    window.search_box.setText("project notes")
    assert wait_for(qapp, lambda: "Loading the search model" in window.status_text.text())
    release.set()
    assert wait_for(qapp, lambda: window.results.count() > 0)
    assert wait_for(qapp, lambda: "indexed" in window.status_text.text())
    finish(window, qapp)


def test_opening_the_window_prepares_search(qapp, engine) -> None:
    window = SearchWindow(engine)
    window.bring_forward()
    assert wait_for(qapp, lambda: engine.embedder.warm_ups == 1)
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
