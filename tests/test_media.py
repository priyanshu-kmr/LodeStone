"""Audio and video: decoding real files, the token budgets, and finding them in search."""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder, FakeOcr, write_video, write_wav
from lodestone.config import MEDIA_SECONDS_MAX, Settings
from lodestone.engine import Engine
from lodestone.media import SAMPLE_RATE, read_audio, read_frames
from lodestone.scanner import estimate_folder, kind_of, parse_exts
from lodestone.search import AUDIO as AUDIO_MODE
from lodestone.search import DOCUMENTS, IMAGES, VIDEOS, parse_query
from lodestone.vectors import AUDIO, VIDEO

RED = (220, 30, 30)
BLUE = (30, 30, 220)


def solid(colour):
    return lambda _time: colour


def frames_of(path, max_seconds=300, max_frames=57, fps=1.0):
    return read_frames(str(path), max_seconds, max_frames, fps, long_side=1024, blank_std=5.0)


@pytest.fixture
def engine(paths, library, fast_settings):
    eng = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
    eng.catalog.add_folder(str(library))
    yield eng
    eng.stop()


def sync(engine: Engine) -> None:
    engine.scan_all()
    while engine.indexer.run_batch():
        pass
    engine.indexer.maintain()


def names(results) -> list[str]:
    return [r.name for r in results]


class TestLimits:
    def test_the_largest_settings_fit_what_the_model_can_take(self) -> None:
        """One embedding holds 8,192 tokens, and going over is an error, not a cut-off."""
        settings = Settings()
        # Every frame could use all of its tokens, plus a marker on each side.
        assert settings.video_max_frames * (settings.video_frame_tokens + 2) <= 8192
        assert MEDIA_SECONDS_MAX * 25 + 16 <= 8192  # Audio costs 25 tokens a second.


class TestReadAudio:
    def test_converts_to_16khz_mono(self, tmp_path) -> None:
        path = tmp_path / "stereo.wav"
        write_wav(path, 2.0, rate=44_100, channels=2)
        samples = read_audio(str(path), 60)
        assert samples.ndim == 1
        assert samples.size == pytest.approx(2 * SAMPLE_RATE, abs=SAMPLE_RATE // 10)
        assert 0.1 < float(abs(samples).max()) < 0.5  # Mixed down, not silenced or clipped.

    def test_stops_at_the_budget(self, tmp_path) -> None:
        path = tmp_path / "long.wav"
        write_wav(path, 5.0)
        assert read_audio(str(path), 1.5).size == int(1.5 * SAMPLE_RATE)

    def test_a_file_without_sound_gives_none(self, tmp_path) -> None:
        path = tmp_path / "quiet.mp4"
        write_video(path, 1, solid(RED))
        assert read_audio(str(path), 60) is None

    def test_reads_the_sound_of_a_video(self, tmp_path) -> None:
        path = tmp_path / "talk.mp4"
        write_video(path, 2, solid(RED), with_sound=True)
        assert read_audio(str(path), 60).size > SAMPLE_RATE

    def test_something_that_is_not_audio_raises(self, tmp_path) -> None:
        path = tmp_path / "fake.mp3"
        path.write_bytes(b"this is not audio at all" * 100)
        with pytest.raises(Exception):
            read_audio(str(path), 60)


class TestReadFrames:
    def test_short_video_gets_one_frame_per_second(self, tmp_path) -> None:
        path = tmp_path / "short.mp4"
        write_video(path, 6, solid(RED))
        assert len(frames_of(path)) == 6

    def test_long_video_is_covered_from_end_to_end(self, tmp_path) -> None:
        """Keyframes are sparse here, so this also checks samples aren't all the same keyframe."""
        path = tmp_path / "long.mp4"
        write_video(path, 40, lambda time: RED if time < 20 else BLUE)
        frames = frames_of(path, max_seconds=40, max_frames=4)
        assert len(frames) == 4
        reds = [f for f in frames if f.resize((1, 1)).getpixel((0, 0))[0] > 120]
        assert len(reds) == 2

    def test_only_the_opening_is_read(self, tmp_path) -> None:
        path = tmp_path / "long.mp4"
        write_video(path, 20, lambda time: RED if time < 10 else BLUE)
        frames = frames_of(path, max_seconds=8)
        assert len(frames) == 8
        assert all(f.resize((1, 1)).getpixel((0, 0))[0] > 120 for f in frames)

    def test_a_long_duration_spreads_the_same_number_of_frames(self, tmp_path) -> None:
        path = tmp_path / "long.mp4"
        write_video(path, 20, lambda time: RED if time < 10 else BLUE)
        frames = frames_of(path, max_seconds=300, max_frames=4)
        assert len(frames) == 4
        assert sum(f.resize((1, 1)).getpixel((0, 0))[0] > 120 for f in frames) == 2

    def test_frames_between_keyframes_are_distinct(self, tmp_path) -> None:
        path = tmp_path / "fade.mp4"
        write_video(path, 4, lambda time: (int(time * 50) + 30, 30, 30))
        reds = [f.resize((1, 1)).getpixel((0, 0))[0] for f in frames_of(path)]
        assert reds == sorted(reds)
        assert len(set(reds)) == 4

    def test_large_frames_are_scaled_down(self, tmp_path) -> None:
        path = tmp_path / "video.mp4"
        write_video(path, 1, solid(RED))
        frame = read_frames(str(path), 1, 1, 1.0, long_side=160, blank_std=5.0)[0]
        assert frame.size == (160, 120)

    def test_blank_frames_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "video.mp4"
        write_video(path, 2, solid(RED))
        assert read_frames(str(path), 2, 2, 1.0, long_side=1024, blank_std=1000.0) == []

    def test_a_file_with_only_sound_has_no_frames(self, tmp_path) -> None:
        path = tmp_path / "sound.webm"
        write_wav(path, 1.0)  # Decoders go by the contents, not the file extension.
        assert frames_of(path) == []


class TestFileRules:
    def test_kinds(self) -> None:
        assert kind_of("song.MP3") == "audio"
        assert kind_of("holiday.MKV") == "video"
        assert kind_of("app.ts") is None  # TypeScript far more often than video.

    def test_they_cannot_be_added_as_extra_types(self) -> None:
        for ext in (".mp3", ".mp4"):
            with pytest.raises(ValueError, match="already handles"):
                parse_exts(ext)

    def test_folder_estimate_counts_them_apart_from_documents(self, library) -> None:
        write_wav(library / "a.wav", 0.1)
        write_video(library / "b.mp4", 1, solid(RED))
        (library / "c.txt").write_text("text")
        estimate = estimate_folder(str(library))
        assert (estimate.documents, estimate.media, estimate.total) == (1, 2, 3)

    def test_query_words_imply_a_filter(self) -> None:
        assert parse_query("video of my dog") .text == "my dog"
        assert parse_query("video of my dog").mode == VIDEOS
        assert parse_query("recording of the meeting").mode == AUDIO_MODE
        assert parse_query("song about rain").text == "rain"


class TestIndexing:
    def test_audio_is_found_through_its_vector(self, engine, library) -> None:
        write_wav(library / "clip.wav", 1.0)
        sync(engine)
        assert engine.vectors.count(AUDIO) == 1
        assert engine.embedder.audio_lengths
        result = engine.search("humming tone")[0]
        assert (result.name, result.kind, result.index_level) == ("clip.wav", "audio", "full")

    def test_video_gets_a_vector_for_pictures_and_one_for_sound(self, engine, library) -> None:
        write_video(library / "clip.mp4", 2, solid(RED), with_sound=True)
        sync(engine)
        assert engine.vectors.count(VIDEO) == 1
        assert engine.vectors.count(AUDIO) == 1
        assert engine.search("red")[0].name == "clip.mp4"
        assert engine.search("humming tone")[0].name == "clip.mp4"
        assert engine.search("humming tone")[0].kind == "video"

    def test_video_without_sound_only_gets_pictures(self, engine, library) -> None:
        write_video(library / "clip.mp4", 2, solid(RED))
        sync(engine)
        assert engine.vectors.count(VIDEO) == 1
        assert engine.vectors.count(AUDIO) == 0
        assert engine.catalog.get_file(str(library / "clip.mp4")).index_level == "full"

    def test_silence_and_tiny_clips_are_found_by_name_only(self, engine, library) -> None:
        write_wav(library / "silent.wav", 1.0, amplitude=0.0)
        write_wav(library / "blip.wav", 0.1)
        sync(engine)
        assert engine.embedder.audio_lengths == []
        for name in ("silent.wav", "blip.wav"):
            assert engine.catalog.get_file(str(library / name)).index_level == "name_only"
        assert names(engine.search("silent")) == ["silent.wav"]

    def test_a_damaged_file_does_not_stop_the_rest(self, engine, library) -> None:
        (library / "broken.mp3").write_bytes(b"not really audio" * 100)
        write_wav(library / "fine.wav", 1.0)
        sync(engine)
        assert engine.catalog.get_file(str(library / "broken.mp3")).status == "error"
        assert engine.catalog.get_file(str(library / "fine.wav")).status == "indexed"
        assert names(engine.search("broken"))[0] == "broken.mp3"

    def test_unreadable_sound_keeps_the_pictures(self, engine, library, monkeypatch) -> None:
        import lodestone.indexer as indexer_module
        from lodestone.media import DecodeError

        def fail(*_args):
            raise DecodeError(0, "unknown codec")

        monkeypatch.setattr(indexer_module, "read_audio", fail)
        write_video(library / "clip.mp4", 2, solid(RED), with_sound=True)
        sync(engine)
        row = engine.catalog.get_file(str(library / "clip.mp4"))
        assert (row.status, row.index_level) == ("indexed", "full")
        assert engine.vectors.count(VIDEO) == 1

    def test_oversized_files_are_found_by_name_without_being_read(self, paths, library, fast_settings) -> None:
        from dataclasses import replace

        settings = replace(fast_settings, max_audio_bytes=10, max_video_bytes=10)
        engine = Engine(paths, FakeEmbedder(), FakeOcr(), settings, watch=False)
        engine.catalog.add_folder(str(library))
        write_wav(library / "big.wav", 1.0)
        write_video(library / "big.mp4", 1, solid(RED))
        sync(engine)
        assert engine.embedder.audio_lengths == [] and engine.embedder.video_lengths == []
        assert engine.catalog.get_file(str(library / "big.wav")).skip_reason == "too_large"
        engine.stop()

    def test_deleting_a_video_removes_both_vectors(self, engine, library) -> None:
        path = library / "clip.mp4"
        write_video(path, 2, solid(RED), with_sound=True)
        sync(engine)
        path.unlink()
        sync(engine)
        assert engine.vectors.count(VIDEO) == 0
        assert engine.vectors.count(AUDIO) == 0

    def test_filters_separate_every_kind(self, engine, library) -> None:
        (library / "red.txt").write_text("red red red")
        write_video(library / "red.mp4", 2, solid(RED))
        write_wav(library / "red.wav", 1.0)
        sync(engine)
        assert names(engine.search("red", mode=VIDEOS)) == ["red.mp4"]
        assert names(engine.search("red", mode=AUDIO_MODE)) == ["red.wav"]
        assert names(engine.search("red", mode=DOCUMENTS)) == ["red.txt"]
        assert names(engine.search("red", mode=IMAGES)) == []


class TestDurationSettings:
    def test_each_duration_limits_its_own_part(self, engine, library) -> None:
        write_video(library / "clip.mp4", 30, solid(RED), with_sound=True)
        engine.set_video_seconds(3)
        engine.set_audio_seconds(10)
        sync(engine)
        assert engine.embedder.video_lengths == [3]
        assert engine.embedder.audio_lengths == [10 * SAMPLE_RATE]

    def test_frames_stay_capped_at_the_longest_duration(self, engine, library) -> None:
        write_video(library / "clip.mp4", 30, solid(RED))
        engine.set_video_seconds(MEDIA_SECONDS_MAX)
        sync(engine)
        assert engine.embedder.video_lengths == [engine.settings.video_max_frames]

    def test_changing_the_video_duration_only_redoes_videos(self, engine, library) -> None:
        write_video(library / "clip.mp4", 4, solid(RED), with_sound=True)
        write_wav(library / "clip.wav", 1.0)
        (library / "note.txt").write_text("a note about something")
        sync(engine)
        documents = engine.embedder.document_calls
        engine.embedder.video_lengths.clear()
        engine.embedder.audio_lengths.clear()
        engine.set_video_seconds(30)
        sync(engine)
        assert len(engine.embedder.video_lengths) == 1
        assert len(engine.embedder.audio_lengths) == 1  # The video's sound, but not the .wav.
        assert engine.embedder.document_calls == documents
        assert engine.vectors.count(VIDEO) == 1  # Replaced, not added to.

    def test_changing_the_audio_duration_also_redoes_the_sound_of_videos(self, engine, library) -> None:
        write_video(library / "clip.mp4", 2, solid(RED), with_sound=True)
        write_wav(library / "clip.wav", 1.0)
        sync(engine)
        engine.embedder.audio_lengths.clear()
        engine.set_audio_seconds(30)
        sync(engine)
        assert len(engine.embedder.audio_lengths) == 2
        assert engine.vectors.count(AUDIO) == 2

    def test_a_duration_of_zero_turns_that_part_off(self, engine, library) -> None:
        write_video(library / "clip.mp4", 2, solid(RED), with_sound=True)
        write_wav(library / "clip.wav", 1.0)
        engine.set_video_seconds(0)
        sync(engine)
        assert engine.embedder.video_lengths == []
        assert engine.vectors.count(AUDIO) == 2  # Sound from both, because its duration is still on.
        assert engine.catalog.get_file(str(library / "clip.mp4")).index_level == "full"
        engine.set_audio_seconds(0)
        sync(engine)
        assert engine.vectors.count(AUDIO) == 0
        for name in ("clip.mp4", "clip.wav"):
            assert engine.catalog.get_file(str(library / name)).index_level == "name_only"

    def test_durations_survive_a_restart(self, paths, library, fast_settings) -> None:
        first = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        first.set_video_seconds(45)
        first.set_audio_seconds(120)
        first.stop()
        second = Engine(paths, FakeEmbedder(), FakeOcr(), fast_settings, watch=False)
        assert (second.video_seconds, second.audio_seconds) == (45, 120)
        second.stop()

    def test_the_text_limit_leaves_media_alone(self, engine, library) -> None:
        write_video(library / "clip.mp4", 2, solid(RED), with_sound=True)
        sync(engine)
        engine.embedder.video_lengths.clear()
        engine.embedder.audio_lengths.clear()
        engine.set_token_limit(1000)
        sync(engine)
        assert engine.embedder.video_lengths == [] and engine.embedder.audio_lengths == []
