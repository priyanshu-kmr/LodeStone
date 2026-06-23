"""End-to-end behaviour of indexing, updates, moves, folders and search, without real models."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from conftest import FakeEmbedder, FakeOcr, textured
from lodestone.engine import Engine
from lodestone.search import IMAGES
from lodestone.vectors import IMAGE, TEXT


@pytest.fixture
def engine(paths, library, fast_settings):
    embedder = FakeEmbedder()
    ocr = FakeOcr()
    eng = Engine(paths, embedder, ocr, fast_settings, watch=False)
    eng.catalog.add_folder(str(library))
    yield eng
    eng.stop()


def sync(engine: Engine) -> None:
    """One deterministic scan plus draining the queue, without background threads."""
    engine.scan_all()
    drain(engine)


def drain(engine: Engine) -> None:
    while engine.indexer.run_batch():
        pass
    engine.indexer.maintain()


def names(results) -> list[str]:
    return [r.name for r in results]


class TestIndexing:
    def test_document_is_searchable_by_meaning_keyword_and_name(self, engine, library) -> None:
        (library / "interview_notes.txt").write_text("We discussed system design and ownership.")
        (library / "recipe.txt").write_text("Boil the pasta and add tomato sauce.")
        sync(engine)
        assert names(engine.search("file which contains system design"))[0] == "interview_notes.txt"
        assert names(engine.search("interview notes"))[0] == "interview_notes.txt"
        assert "tomato" in engine.search("tomato sauce")[0].snippet

    def test_image_is_searchable_through_its_vector(self, engine, library) -> None:
        textured((640, 480), (220, 20, 20)).save(library / "a.png")
        textured((640, 480), (20, 20, 220)).save(library / "b.png")
        sync(engine)
        assert names(engine.search("image of red", mode=IMAGES))[0] == "a.png"
        assert names(engine.search("photo of blue"))[0] == "b.png"

    def test_image_filter_hides_documents(self, engine, library) -> None:
        (library / "red.txt").write_text("red red red")
        textured((640, 480), (220, 20, 20)).save(library / "a.png")
        sync(engine)
        assert names(engine.search("image of red")) == ["a.png"]

    def test_ocr_text_is_searchable(self, engine, library) -> None:
        engine.indexer._ocr.text = "Invoice number 48213"
        textured((640, 480), (20, 200, 20)).save(library / "scan.png")
        sync(engine)
        result = engine.search("48213")[0]
        assert result.name == "scan.png"
        assert "keyword" in result.matched_by
        assert "48213" in result.snippet

    def test_low_quality_images_stay_findable_by_name_only(self, engine, library) -> None:
        textured((100, 100), (10, 200, 10)).save(library / "tiny_logo.png")
        scene = Image.linear_gradient("L").resize((640, 480)).convert("RGB")
        scene.filter(ImageFilter.GaussianBlur(25)).save(library / "blurry_shot.png")
        Image.new("RGB", (640, 480), (90, 90, 90)).save(library / "blank_page.png")
        (library / "broken_photo.jpg").write_bytes(b"not an image")
        sync(engine)
        assert engine.indexer._ocr.calls == 0
        assert engine.embedder.image_calls == 0
        assert engine.vectors.count(IMAGE) == 0
        reasons = {r.name: r.skip_reason for r in engine.name_only_images()}
        assert reasons == {
            "tiny_logo.png": "small",
            "blurry_shot.png": "blurry",
            "blank_page.png": "blank",
            "broken_photo.jpg": "unreadable",
        }
        assert names(engine.search("tiny logo"))[0] == "tiny_logo.png"

    def test_index_anyway_promotes_an_image(self, engine, library) -> None:
        path = library / "tiny.png"
        textured((100, 100), (220, 10, 10)).save(path)
        sync(engine)
        assert engine.vectors.count(IMAGE) == 0
        engine.index_anyway(str(path))
        drain(engine)
        assert engine.vectors.count(IMAGE) == 1
        assert engine.catalog.get_file(str(path)).index_level == "full"
        # The choice survives a later scan.
        sync(engine)
        assert engine.catalog.get_file(str(path)).index_level == "full"

    def test_unsupported_office_format_is_name_only(self, engine, library) -> None:
        (library / "old_report.doc").write_bytes(b"\xd0\xcf\x11\xe0binary")
        sync(engine)
        row = engine.catalog.get_file(str(library / "old_report.doc"))
        assert row.index_level == "name_only" and row.skip_reason == "unsupported"
        assert names(engine.search("old report"))[0] == "old_report.doc"

    def test_one_bad_file_does_not_stop_the_run(self, engine, library) -> None:
        (library / "bad.pdf").write_bytes(b"%PDF-broken")
        (library / "good.txt").write_text("still indexed")
        sync(engine)
        assert engine.catalog.get_file(str(library / "bad.pdf")).status == "error"
        assert engine.catalog.get_file(str(library / "good.txt")).status == "indexed"
        assert engine.catalog.queue_size() == 0

    def test_failed_files_are_retried_on_the_next_start(self, engine, library) -> None:
        engine.embedder.fail_on_size = (321, 321)
        path = library / "flaky.png"
        textured((321, 321), (200, 10, 10)).save(path)
        sync(engine)
        assert engine.catalog.get_file(str(path)).status == "error"
        # A rescan leaves it alone, so a broken file isn't retried over and over.
        sync(engine)
        assert engine.catalog.get_file(str(path)).status == "error"

        engine.embedder.fail_on_size = None
        assert engine.catalog.retry_failed() == 1
        drain(engine)
        row = engine.catalog.get_file(str(path))
        assert row.status == "indexed" and row.index_level == "full"
        assert engine.vectors.count(IMAGE) == 1

    def test_failed_image_batch_only_fails_the_bad_image(self, engine, library) -> None:
        engine.embedder.fail_on_size = (321, 321)
        textured((321, 321), (200, 10, 10)).save(library / "bad.png")
        textured((640, 480), (10, 10, 200)).save(library / "good.png")
        sync(engine)
        assert engine.catalog.get_file(str(library / "bad.png")).status == "error"
        assert engine.catalog.get_file(str(library / "good.png")).status == "indexed"


class TestRanking:
    def test_only_the_opening_is_embedded_but_all_of_it_is_keyword_searchable(self, engine, library) -> None:
        rows = "\n".join(f"row number {i} with value item{i} and some padding text" for i in range(2000))
        (library / "big.csv").write_text("id,value\n" + rows)
        sync(engine)
        # 500 tokens at 4 characters each is 2,000 characters, which spans two 1,200-character chunks.
        assert engine.vectors.count(TEXT) == 2
        assert names(engine.search("item1999"))[0] == "big.csv"

    def test_files_indexed_before_the_limit_are_trimmed_once(self, paths, library) -> None:
        from lodestone.config import Settings

        (library / "long.txt").write_text("\n\n".join(f"paragraph {i} " + "words " * 150 for i in range(30)))
        roomy = Settings(debounce_seconds=0, delete_delay_seconds=0, embed_token_limit=100_000)
        first = Engine(paths, FakeEmbedder(), FakeOcr(), roomy, watch=False)
        first.catalog.add_folder(str(library))
        sync(first)
        assert first.vectors.count(TEXT) > 5
        first.catalog.close()

        strict = Settings(debounce_seconds=0, delete_delay_seconds=0)
        second = Engine(paths, FakeEmbedder(), FakeOcr(), strict, watch=False)
        sync(second)
        assert second.vectors.count(TEXT) == 3  # Chunks of about 900 characters, to reach 2,000.
        assert second.embedder.document_calls == 0  # Trimming needs no re-embedding.
        assert second.catalog.get_meta("embed_token_limit") == "500"
        assert names(second.search("paragraph 29"))[0] == "long.txt"

    def test_scanned_pdf_stops_ocr_once_it_has_enough_text(self, engine, library) -> None:
        import pymupdf

        engine.indexer._ocr.text = "scanned exam question about graphs " * 40  # About 1,400 characters.
        stamp = library.parent / "stamp.png"
        textured((300, 300), (120, 120, 120)).save(stamp)
        doc = pymupdf.open()
        for _ in range(10):
            doc.new_page().insert_image(pymupdf.Rect(0, 0, 300, 300), filename=str(stamp))
        doc.save(library / "scan.pdf")
        doc.close()
        sync(engine)
        assert engine.indexer._ocr.calls == 2  # Two pages reach 2,000 characters; ten were available.
        assert engine.catalog.get_file(str(library / "scan.pdf")).index_level == "full"

    def test_very_short_text_is_keyword_only(self, engine, library) -> None:
        (library / "tiny.txt").write_text("ok 42")
        sync(engine)
        assert engine.vectors.count(TEXT) == 0
        assert names(engine.search("tiny"))[0] == "tiny.txt"

    def test_keyword_match_on_one_of_many_words_does_not_win(self, engine, library) -> None:
        (library / "note.txt").write_text("the night shift starts at nine for everyone")
        textured((640, 480), (220, 20, 20)).save(library / "shot.png")
        textured((640, 480), (20, 20, 220)).save(library / "other.png")
        sync(engine)
        # The document shares one word out of three, and the red image matches by meaning.
        results = engine.search("red rocket night")
        assert "keyword" not in results[0].matched_by

    def test_exact_filename_beats_a_meaning_match(self, engine, library) -> None:
        textured((100, 100), (10, 200, 10)).save(library / "tiny_badge.png")
        textured((640, 480), (220, 20, 20)).save(library / "photo.png")
        (library / "story.txt").write_text("a long story about nothing in particular at all")
        sync(engine)
        assert names(engine.search("tiny badge"))[0] == "tiny_badge.png"

    def test_snippet_shows_the_matching_passage_not_the_filename(self, engine, library) -> None:
        (library / "tomato.txt").write_text("Boil the pasta and add tomato sauce.")
        sync(engine)
        result = engine.search("tomato")[0]
        assert result.name == "tomato.txt"
        assert "sauce" in result.snippet


class TestLargeFiles:
    def test_huge_spreadsheet_only_reads_its_opening_rows(self, engine, library) -> None:
        from openpyxl import Workbook

        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet("Sales")
        sheet.append(["invoice", "product"])
        for i in range(3000):
            sheet.append([i, f"product{i}"])
        path = library / "sales.xlsx"
        workbook.save(path)

        started = time.perf_counter()
        sync(engine)
        assert time.perf_counter() - started < 60
        assert engine.catalog.get_file(str(path)).status == "indexed"
        assert names(engine.search("product10"))[0] == "sales.xlsx"
        # Rows beyond the limit aren't read.
        assert engine.search("product2999") == [] or "product2999" not in engine.search("product2999")[0].snippet
        assert engine.vectors.count(TEXT) <= 2

    def test_office_file_over_the_limit_is_name_only(self, paths, library) -> None:
        from lodestone.config import Settings

        settings = Settings(debounce_seconds=0, delete_delay_seconds=0, max_office_bytes=10)
        eng = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        eng.catalog.add_folder(str(library))
        (library / "big_report.docx").write_bytes(b"x" * 100)
        sync(eng)
        row = eng.catalog.get_file(str(library / "big_report.docx"))
        assert row.index_level == "name_only" and row.skip_reason == "too_large"
        assert names(eng.search("big report"))[0] == "big_report.docx"
        eng.stop()

    def test_stuck_conversion_fails_that_file_and_the_queue_moves_on(self, engine, library, monkeypatch) -> None:
        import lodestone.extract as extract_module
        from lodestone.office import OfficeWorker
        from office_helpers import hang_forever

        worker = OfficeWorker(timeout=2, handler=hang_forever)
        monkeypatch.setattr(extract_module, "shared_worker", lambda: worker)
        (library / "stuck.docx").write_bytes(b"x")
        (library / "fine.txt").write_text("this one still gets indexed")
        try:
            sync(engine)
            row = engine.catalog.get_file(str(library / "stuck.docx"))
            assert row.status == "error" and "gave up" in row.error
            assert engine.catalog.get_file(str(library / "fine.txt")).status == "indexed"
            assert engine.catalog.queue_size() == 0
            assert names(engine.search("stuck"))[0] == "stuck.docx"
        finally:
            worker.close()


class TestGpuSetting:
    def test_off_by_default_and_remembered(self, paths, fast_settings) -> None:
        first = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        assert first.use_gpu is False
        first.set_use_gpu(True)
        assert first.embedder.gpu_settings == [True]
        first.catalog.close()
        second = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        assert second.use_gpu is True


class TestIdle:
    def test_idle_frees_ocr_and_the_indexing_model(self, paths, library) -> None:
        from lodestone.config import Settings

        settings = Settings(debounce_seconds=0, delete_delay_seconds=0, vision_idle_unload_seconds=0.1)
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        engine.catalog.add_folder(str(library))
        (library / "a.txt").write_text("some text worth embedding")
        sync(engine)
        engine.indexer.idle()
        assert engine.embedder.released == 0  # Not idle for long enough yet.
        time.sleep(0.2)
        engine.indexer.idle()
        assert engine.embedder.released == 1
        assert engine.indexer._ocr.released is True
        engine.stop()


class TestBusyFiles:
    def _engine(self, paths, library, cooldown: float) -> Engine:
        from lodestone.config import Settings

        settings = Settings(debounce_seconds=0, delete_delay_seconds=0, reindex_cooldown_seconds=cooldown)
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        engine.catalog.add_folder(str(library))
        return engine

    def test_file_that_keeps_changing_is_not_re_embedded_during_the_cooldown(self, paths, library) -> None:
        engine = self._engine(paths, library, cooldown=600)
        path = library / "app_debug.log"
        path.write_text("first line of the log file")
        sync(engine)
        calls = engine.embedder.document_calls

        for i in range(3):
            path.write_text(f"first line of the log file\nanother line {i}")
            os.utime(path, (time.time() + 5 + i, time.time() + 5 + i))
            sync(engine)
        assert engine.embedder.document_calls == calls
        # It's still waiting, but that doesn't count as indexing.
        assert engine.catalog.queue_size() == 1
        assert engine.status().state == "Up to date"
        assert engine.status().queued == 0
        engine.stop()

    def test_changes_are_picked_up_after_the_cooldown(self, paths, library) -> None:
        engine = self._engine(paths, library, cooldown=0.5)
        path = library / "app_debug.log"
        path.write_text("the original log contents")
        sync(engine)
        path.write_text("completely replaced log contents")
        os.utime(path, (time.time() + 5, time.time() + 5))
        sync(engine)
        assert "completely" not in engine.catalog.chunks_for_file(engine.catalog.get_file(str(path)).id)[-1][2]
        time.sleep(0.6)
        drain(engine)
        assert "completely" in engine.catalog.chunks_for_file(engine.catalog.get_file(str(path)).id)[-1][2]
        engine.stop()

    def test_waiting_files_do_not_stop_the_idle_cleanup(self, paths, library) -> None:
        from lodestone.config import Settings

        settings = Settings(
            debounce_seconds=0, delete_delay_seconds=0, reindex_cooldown_seconds=600,
            vision_idle_unload_seconds=0.2,
        )
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        engine.catalog.add_folder(str(library))
        path = library / "app_debug.log"
        path.write_text("first line of the log file")
        sync(engine)
        time.sleep(0.3)
        path.write_text("first line of the log file, then more")
        os.utime(path, (time.time() + 5, time.time() + 5))
        sync(engine)  # Only hits the cooldown, so this isn't real work.
        engine.indexer.idle()
        assert engine.embedder.released == 1
        engine.stop()


class TestUpdates:
    def test_unchanged_files_are_not_reprocessed(self, engine, library) -> None:
        (library / "a.txt").write_text("hello world")
        sync(engine)
        calls = engine.embedder.document_calls
        sync(engine)
        assert engine.embedder.document_calls == calls
        assert engine.vectors.count(TEXT) == 1

    def test_changed_file_replaces_old_content(self, engine, library) -> None:
        path = library / "a.txt"
        path.write_text("alpha alpha")
        sync(engine)
        path.write_text("omega omega")
        os.utime(path, (time.time() + 5, time.time() + 5))
        sync(engine)
        assert engine.vectors.count(TEXT) == 1
        assert engine.search("alpha") == [] or "omega" in engine.search("alpha")[0].snippet
        assert "omega" in engine.search("omega")[0].snippet

    def test_touched_file_is_not_re_embedded(self, engine, library) -> None:
        path = library / "a.txt"
        path.write_text("same content")
        sync(engine)
        calls = engine.embedder.document_calls
        os.utime(path, (time.time() + 5, time.time() + 5))
        sync(engine)
        assert engine.embedder.document_calls == calls

    def test_deleted_file_leaves_the_index(self, engine, library) -> None:
        path = library / "a.txt"
        path.write_text("goodbye")
        sync(engine)
        path.unlink()
        sync(engine)
        assert engine.catalog.get_file(str(path)) is None
        assert engine.vectors.count(TEXT) == 0
        assert engine.search("goodbye") == []

    def test_moved_file_keeps_its_vectors(self, engine, library) -> None:
        old = library / "old_name.png"
        textured((640, 480), (220, 20, 20)).save(old)
        sync(engine)
        file_id = engine.catalog.get_file(str(old)).id
        calls = engine.embedder.image_calls
        new_dir = library / "moved"
        new_dir.mkdir()
        new = new_dir / "new_name.png"
        old.rename(new)
        sync(engine)
        assert engine.embedder.image_calls == calls
        row = engine.catalog.get_file(str(new))
        assert row is not None and row.id == file_id
        assert engine.catalog.get_file(str(old)) is None
        assert engine.vectors.count(IMAGE) == 1
        assert names(engine.search("new name"))[0] == "new_name.png"
        assert names(engine.search("old name")) == [] or names(engine.search("old name"))[0] != "old_name.png"

    def test_copy_is_indexed_separately(self, engine, library) -> None:
        (library / "a.txt").write_text("copy me please")
        sync(engine)
        (library / "b.txt").write_text("copy me please")
        sync(engine)
        assert engine.catalog.get_file(str(library / "a.txt")) is not None
        assert engine.catalog.get_file(str(library / "b.txt")) is not None
        assert engine.vectors.count(TEXT) == 2

    def test_stale_result_is_hidden_and_cleaned_up(self, engine, library) -> None:
        path = library / "a.txt"
        path.write_text("vanishing")
        sync(engine)
        path.unlink()
        assert engine.search("vanishing") == []
        drain(engine)
        assert engine.catalog.get_file(str(path)) is None


class TestCrashRecovery:
    def test_interrupted_file_is_redone_without_duplicates(self, paths, library, fast_settings) -> None:
        path = library / "a.txt"
        path.write_text("crash test content")

        first = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        first.catalog.add_folder(str(library))
        first.scan_all()
        original_finish = first.catalog.finish_file

        def crash(*args, **kwargs):
            raise KeyboardInterrupt

        first.catalog.finish_file = crash
        with pytest.raises(KeyboardInterrupt):
            first.indexer.run_batch()
        first.catalog.finish_file = original_finish
        # Vectors were written, but the file was never marked as indexed.
        assert first.catalog.get_file(str(path)).status == "pending"
        assert first.catalog.queue_size() == 1
        assert first.vectors.count(TEXT) == 1
        first.catalog.close()

        second = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        sync(second)
        assert second.catalog.get_file(str(path)).status == "indexed"
        assert second.vectors.count(TEXT) == 1
        assert names(second.search("crash test"))[0] == "a.txt"


class TestFolders:
    def test_remove_folder_forgets_entries_but_keeps_files(self, engine, library) -> None:
        path = library / "a.txt"
        path.write_text("keep the file")
        sync(engine)
        engine.remove_folder(str(library))
        drain(engine)
        assert path.exists()
        assert engine.catalog.get_file(str(path)) is None
        assert engine.vectors.count(TEXT) == 0

    def test_removing_a_nested_folder_keeps_what_the_parent_covers(self, engine, library) -> None:
        sub = library / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("nested")
        engine.catalog.add_folder(str(sub))
        sync(engine)
        engine.remove_folder(str(sub))
        drain(engine)
        assert engine.catalog.get_file(str(sub / "a.txt")) is not None

    def test_overlapping_folders_store_each_file_once(self, engine, library) -> None:
        sub = library / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("only once")
        engine.catalog.add_folder(str(sub))
        sync(engine)
        assert engine.vectors.count(TEXT) == 1

    def test_exclude_removes_and_skips(self, engine, library) -> None:
        sub = library / "private"
        sub.mkdir()
        (sub / "a.txt").write_text("secret")
        sync(engine)
        engine.add_exclude(str(sub))
        sync(engine)
        assert engine.catalog.get_file(str(sub / "a.txt")) is None
        (sub / "b.txt").write_text("also secret")
        sync(engine)
        assert engine.catalog.get_file(str(sub / "b.txt")) is None
        engine.remove_exclude(str(sub))
        sync(engine)
        assert engine.catalog.get_file(str(sub / "b.txt")) is not None

    def test_turning_a_folder_off_hides_results_but_keeps_the_index(self, engine, library) -> None:
        (library / "a.txt").write_text("sleeping content")
        sync(engine)
        calls = engine.embedder.document_calls
        engine.set_folder_enabled(str(library), False)
        sync(engine)
        assert engine.search("sleeping content") == []
        assert engine.vectors.count(TEXT) == 1
        engine.set_folder_enabled(str(library), True)
        sync(engine)
        assert names(engine.search("sleeping content")) == ["a.txt"]
        assert engine.embedder.document_calls == calls

    def test_missing_root_does_not_wipe_the_index(self, engine, library, tmp_path) -> None:
        (library / "a.txt").write_text("on a drive that gets unplugged")
        sync(engine)
        library.rename(tmp_path / "unplugged")
        sync(engine)
        assert engine.catalog.get_file(str(library / "a.txt")) is not None


class TestDatasetPrompt:
    def _make_dataset(self, library: Path, count: int = 30) -> Path:
        project = library / "project"
        (project / ".git").mkdir(parents=True)
        data = project / "data"
        data.mkdir()
        for i in range(count):
            textured((64, 64), (200, 10, 10), seed=i).save(data / f"{i:05d}.png")
        return data

    def _engine(self, paths, library) -> Engine:
        from lodestone.config import Settings

        settings = Settings(debounce_seconds=0, delete_delay_seconds=0, dataset_min_images=20)
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        engine.catalog.add_folder(str(library))
        return engine

    def test_pauses_until_answered_then_skip(self, paths, library) -> None:
        data = self._make_dataset(library)
        (library / "note.txt").write_text("normal file")
        engine = self._engine(paths, library)
        sync(engine)
        prompts = engine.dataset_prompts()
        assert [Path(p.path).name for p in prompts] == ["data"]
        assert engine.catalog.get_file(str(data / "00000.png")) is None
        assert engine.catalog.get_file(str(library / "note.txt")) is not None

        engine.decide_dataset(str(data), index_it=False)
        sync(engine)
        assert engine.dataset_prompts() == []
        assert engine.catalog.get_file(str(data / "00000.png")) is None
        assert str(data) in engine.excludes()

    def test_index_anyway_is_remembered(self, paths, library) -> None:
        data = self._make_dataset(library)
        engine = self._engine(paths, library)
        sync(engine)
        engine.decide_dataset(str(data), index_it=True)
        sync(engine)
        assert engine.catalog.get_file(str(data / "00000.png")) is not None
        sync(engine)
        assert engine.dataset_prompts() == []


class TestSeeding:
    def test_defaults_are_seeded_once(self, paths, fast_settings, monkeypatch, library) -> None:
        import lodestone.engine as module

        monkeypatch.setattr(module, "default_folders", lambda: [library])
        monkeypatch.setattr(module, "seed_excludes", lambda: [library / "skip"])
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        engine.seed_defaults()
        assert [f.path for f in engine.folders()] == [str(library)]
        assert engine.excludes() == [str(library / "skip")]
        engine.remove_folder(str(library))
        engine.seed_defaults()
        assert engine.folders() == []


class TestLiveWatching:
    def test_new_changed_and_deleted_files_are_picked_up(self, paths, library) -> None:
        from lodestone.config import Settings

        settings = Settings(debounce_seconds=0.2, delete_delay_seconds=0.2)
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=True)
        engine.catalog.add_folder(str(library))
        engine.start()
        try:
            assert engine.wait_until_idle(30)
            path = library / "new_note.txt"
            path.write_text("freshly saved content")
            assert _eventually(lambda: engine.search("freshly saved"), 15)
            assert names(engine.search("freshly saved"))[0] == "new_note.txt"
            path.unlink()
            assert _eventually(lambda: engine.catalog.get_file(str(path)) is None, 15)
        finally:
            engine.stop()


def _eventually(check, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.1)
    return False


def test_keyword_search_stays_in_sync_after_delete(engine, library) -> None:
    path = library / "a.txt"
    path.write_text("zebra stripes")
    sync(engine)
    assert engine.catalog.keyword_search("zebra", 5)
    path.unlink()
    sync(engine)
    assert engine.catalog.keyword_search("zebra", 5) == []
    conn = sqlite3.connect(engine.paths.catalog)
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    conn.close()
