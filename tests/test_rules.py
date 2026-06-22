"""Quality gate, folder rules, dataset check, chunking and query parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from conftest import textured
from lodestone import quality
from lodestone.catalog import fts_query, name_chunk_text
from lodestone.config import Settings, is_under
from lodestone.extract import chunk_text, read_text_file, within_embed_limit
from lodestone.ocr import clean_lines
from lodestone.scanner import find_datasets, is_cloud_only, is_temp_file, kind_of, path_allowed, scan_root, walk
from lodestone.search import ALL, DOCUMENTS, IMAGES, parse_query, rrf_merge, standout_scores


def scene() -> Image.Image:
    """Shapes on a colour gradient: real structure that blur can remove."""
    ramp = np.linspace(0, 255, 640).astype(np.uint8)
    pixels = np.stack([np.tile(ramp, (480, 1))] * 3, axis=2)
    pixels[100:300, 150:400] = (250, 40, 40)
    return Image.fromarray(pixels)


class TestQuality:
    def test_small_only_when_both_sides_are_under_the_limit(self, tmp_path: Path) -> None:
        small = tmp_path / "icon.png"
        textured((150, 150), (200, 10, 10)).save(small)
        assert quality.assess(str(small)).reason == quality.SMALL

        wide = tmp_path / "banner.png"
        textured((800, 150), (200, 10, 10)).save(wide)
        assert quality.assess(str(wide)).reason == quality.FULL

    def test_corrupt_file_is_unreadable(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"this is not an image")
        assert quality.assess(str(broken)).reason == quality.UNREADABLE

    def test_blank_image(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank.png"
        Image.new("RGB", (640, 480), (128, 128, 128)).save(blank)
        assert quality.assess(str(blank)).reason == quality.BLANK

    def test_heavily_blurred_image_is_blurry(self, tmp_path: Path) -> None:
        blurred = tmp_path / "blurred.png"
        scene().filter(ImageFilter.GaussianBlur(25)).save(blurred)
        result = quality.assess(str(blurred))
        assert result.reason == quality.BLURRY
        assert result.sharpness is not None

    def test_small_sharp_subject_on_plain_background_is_not_blank(self, tmp_path: Path) -> None:
        image = Image.new("RGB", (640, 480), (120, 120, 120))
        image.paste(textured((60, 60), (120, 120, 120), seed=3), (290, 210))
        path = tmp_path / "subject.png"
        image.save(path)
        assert quality.assess(str(path)).reason == quality.FULL

    def test_sharp_subject_on_blurred_background_passes(self, tmp_path: Path) -> None:
        image = scene().filter(ImageFilter.GaussianBlur(25))
        image.paste(textured((120, 120), (120, 120, 120), seed=3), (260, 180))
        path = tmp_path / "bokeh.png"
        image.save(path)
        assert quality.assess(str(path)).reason == quality.FULL

    def test_index_anyway_overrides_the_rules(self, tmp_path: Path) -> None:
        small = tmp_path / "icon.png"
        textured((50, 50), (0, 200, 0)).save(small)
        result = quality.assess(str(small), force=True)
        assert result.passed and result.image is not None

    def test_transparent_image_is_flattened_not_blank(self, tmp_path: Path) -> None:
        pixels = np.zeros((480, 640, 4), dtype=np.uint8)
        pixels[100:380, 100:540] = (0, 0, 0, 255)
        path = tmp_path / "logo.png"
        Image.fromarray(pixels, "RGBA").save(path)
        result = quality.assess(str(path))
        assert result.reason == quality.FULL
        assert result.image.mode == "RGB"


class TestFolderRules:
    def test_kinds(self) -> None:
        assert kind_of("notes.PDF") == "document"
        assert kind_of("photo.JPG") == "image"
        assert kind_of("old.doc") == "name_only"
        assert kind_of("setup.exe") is None

    def test_temp_files(self) -> None:
        assert is_temp_file("~$report.docx")
        assert is_temp_file("movie.crdownload")
        assert not is_temp_file("report.docx")

    def test_cloud_only_attribute(self) -> None:
        assert is_cloud_only(0x00400000)
        assert not is_cloud_only(0x20)

    def test_is_under_ignores_case_and_partial_names(self) -> None:
        assert is_under(r"C:\Users\A\Docs\x.txt", r"c:\users\a\docs")
        assert not is_under(r"C:\Users\A\Docs2\x.txt", r"C:\Users\A\Docs")

    def test_path_allowed(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        roots = [str(root)]
        assert path_allowed(str(root / "a.txt"), roots, [])
        assert not path_allowed(str(root / "node_modules" / "a.txt"), roots, [])
        assert not path_allowed(str(root / "secret.noindex" / "a.txt"), roots, [])
        assert not path_allowed(str(root / "skip" / "a.txt"), roots, [str(root / "skip")])
        assert not path_allowed(str(tmp_path / "elsewhere" / "a.txt"), roots, [])

    def test_walk_skips_rules_and_virtualenvs(self, tmp_path: Path) -> None:
        (tmp_path / "keep").mkdir()
        (tmp_path / "keep" / "a.txt").write_text("x")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "b.txt").write_text("x")
        (tmp_path / "myenv").mkdir()
        (tmp_path / "myenv" / "pyvenv.cfg").write_text("x")
        (tmp_path / "myenv" / "c.txt").write_text("x")
        (tmp_path / "keep" / "ignored.exe").write_text("x")
        files, _ = walk(str(tmp_path))
        assert [Path(f.path).name for f in files] == ["a.txt"]


def _images(folder: Path, count: int, name: str = "{:05d}.png", size: int = 2000) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / name.format(i)).write_bytes(b"x" * size)


class TestDatasetCheck:
    def settings(self) -> Settings:
        return Settings(dataset_min_images=50)

    def test_finds_the_real_folder_not_its_parents(self, tmp_path: Path) -> None:
        root = tmp_path / "Desktop"
        project = root / "codes" / "ML" / "wildlife"
        (project / ".git").mkdir(parents=True)
        for sub in ("Angle", "Sensor"):
            _images(project / "Elephant" / sub, 40)
        result = scan_root(str(root), settings=self.settings())
        assert [Path(c.path).name for c in result.candidates] == ["Elephant"]
        assert "code nearby" in result.candidates[0].signs

    def test_holiday_photos_are_not_flagged(self, tmp_path: Path) -> None:
        root = tmp_path / "Pictures"
        _images(root / "Holiday", 100, "IMG_{:04d}.jpg", size=200_000)
        result = scan_root(str(root), settings=self.settings())
        assert result.candidates == []

    def test_watched_folder_itself_is_never_flagged(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        _images(root, 100)
        (root / "src").mkdir()
        _, top = walk(str(root))
        assert find_datasets(top, self.settings()) == []

    def test_too_few_images_is_not_flagged(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        _images(root / "small", 10)
        assert scan_root(str(root), settings=self.settings()).candidates == []


class TestText:
    def test_chunks_overlap_and_cover_everything(self) -> None:
        settings = Settings(chunk_chars=100, chunk_overlap=20)
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, settings)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)
        assert "word0" in chunks[0] and "word199" in chunks[-1]

    def test_chunk_cap(self) -> None:
        settings = Settings(chunk_chars=50, chunk_overlap=5, max_chunks_per_file=3)
        assert len(chunk_text("x " * 1000, settings)) == 3

    def test_empty_text(self) -> None:
        assert chunk_text("   \n  ") == []

    def test_embed_limit_cuts_at_the_token_budget(self) -> None:
        settings = Settings(embed_token_limit=10, chars_per_token=4)
        assert within_embed_limit(["a" * 30, "b" * 30, "c" * 30], settings) == ["a" * 30, "b" * 10]
        assert within_embed_limit(["short"], settings) == ["short"]
        assert within_embed_limit([], settings) == []

    def test_reads_common_encodings(self, tmp_path: Path) -> None:
        for encoding, sample in (("utf-8", "café"), ("utf-16", "café"), ("cp1252", "café")):
            path = tmp_path / f"{encoding}.txt"
            path.write_text(sample, encoding=encoding)
            assert read_text_file(str(path)) == sample

    def test_name_chunk_splits_words(self) -> None:
        text = name_chunk_text(r"C:\work\Job Search\interview_notes-final.pdf")
        assert "interview notes final" in text and "pdf" in text and "Job Search" in text
        assert "Interview Notes" in name_chunk_text(r"C:\x\InterviewNotes.txt")

    def test_fts_query_is_safe(self) -> None:
        assert fts_query('notes" OR (bad*') == '"notes" OR "bad"'
        assert fts_query("the notes of my") == '"notes"'
        assert fts_query("of the") == '"of" OR "the"'
        assert fts_query("!!!") == ""


class TestQueryParsing:
    def test_examples_from_the_brief(self) -> None:
        parsed = parse_query("file which contains my interview notes")
        assert parsed == type(parsed)("my interview notes", ALL)
        parsed = parse_query("image which has an owl sitting next to a table lamp with one leg")
        assert parsed.mode == IMAGES
        assert parsed.text == "an owl sitting next to a table lamp with one leg"

    def test_filters(self) -> None:
        assert parse_query("photo of my dog").mode == IMAGES
        assert parse_query("find me a screenshot of the error").mode == IMAGES
        assert parse_query("pdf about taxes") == type(parse_query("x"))("taxes", DOCUMENTS)

    def test_plain_query_is_untouched(self) -> None:
        assert parse_query("  interview   notes ").text == "interview notes"
        assert parse_query("photosynthesis notes").mode == ALL

    def test_bare_word_is_kept(self) -> None:
        parsed = parse_query("photo")
        assert parsed.text == "photo" and parsed.mode == ALL

    def test_rrf_rewards_agreement(self) -> None:
        scores, matched = rrf_merge({"a": [1, 2], "b": [2, 3]}, k=60)
        assert max(scores, key=scores.get) == 2
        assert matched[2] == ["a", "b"]

    def test_rrf_weights(self) -> None:
        scores, _ = rrf_merge({"a": [1], "b": [2]}, k=60, weights={"b": {2: 2.0}})
        assert scores[2] == 2 * scores[1]

    def test_standout_score_finds_the_clear_match(self) -> None:
        clear = standout_scores({1: 0.75, 2: 0.60, 3: 0.59, 4: 0.58, 5: 0.57}, 0.02)
        flat = standout_scores({6: 0.63, 7: 0.62, 8: 0.62, 9: 0.61, 10: 0.61}, 0.02)
        assert clear[1] > flat[6]
        assert standout_scores({}, 0.02) == {}


class TestLoadGate:
    def test_loading_waits_for_work_and_work_waits_for_loading(self) -> None:
        import threading
        import time

        from lodestone.embedder import _LoadGate

        gate = _LoadGate()
        events: list[str] = []

        def load() -> None:
            with gate.loading():
                events.append("load start")
                time.sleep(0.1)
                events.append("load end")

        def work() -> None:
            with gate.working():
                events.append("second work")

        with gate.working():
            loader = threading.Thread(target=load)
            loader.start()
            time.sleep(0.05)
            events.append("first work ends")
        time.sleep(0.03)  # The load is now in progress.
        worker = threading.Thread(target=work)
        worker.start()
        loader.join()
        worker.join()
        assert events == ["first work ends", "load start", "load end", "second work"]


class TestOcrCleanup:
    def test_stray_marks_from_photos_are_dropped(self) -> None:
        assert clean_lines(["O"], [0.89]).text == ""
        assert clean_lines(["8"], [0.9]).text == ""
        assert clean_lines(["0000"], [0.8]).text == ""
        assert clean_lines([], []).score is None

    def test_real_text_is_kept(self) -> None:
        result = clean_lines(["INVOICE number 48213", "O", "48213"], [0.99, 0.9, 0.95])
        assert result.text == "INVOICE number 48213\n48213"
        assert result.score == pytest.approx(0.97)

    def test_low_confidence_text_is_dropped(self) -> None:
        assert clean_lines(["blurry words"], [0.4]).text == ""
