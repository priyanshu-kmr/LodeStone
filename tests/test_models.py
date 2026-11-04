"""Checks against the real model and OCR. Slow and needs downloads, so run with:
pytest -m models --override-ini=addopts=
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from lodestone.config import DEFAULT_MODEL, MEDIA_SECONDS_MAX, MODEL_ID, Settings

pytestmark = pytest.mark.models


def _photo(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).resize((400, 300))


@pytest.fixture(scope="module")
def embedder():
    from lodestone.embedder import GemmaEmbedder

    return GemmaEmbedder()


def test_cpu_mode_never_touches_the_gpu() -> None:
    """In a fresh process, because other tests in this file may start the GPU."""
    code = (
        "import torch\n"
        "from PIL import Image\n"
        "from lodestone.embedder import GemmaEmbedder\n"
        "e = GemmaEmbedder()\n"
        "e.encode_query('hello')\n"
        "e.encode_documents(['some document text for the test'])\n"
        "e.encode_images([Image.new('RGB', (300, 300), (200, 100, 50))])\n"
        "assert not torch.cuda.is_initialized(), 'CUDA was started'\n"
        "assert {str(p.device) for p in e._query_model.parameters()} == {'cpu'}\n"
        "assert {str(p.device) for p in e._index_model.parameters()} == {'cpu'}\n"
        "assert torch.get_num_threads() == 4\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-1500:]


def test_models_loading_at_the_same_time_all_keep_one_dtype(embedder) -> None:
    """The app's startup can load both models in parallel, which once broke every image."""
    import torch

    errors: list[Exception] = []

    def run(work) -> None:
        try:
            work()
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(embedder.warm_up,)),
        threading.Thread(target=run, args=(lambda: embedder.encode_images([_photo(0)]),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []

    # On the CPU, search uses bfloat16 to save RAM, and indexing uses float32 for speed.
    assert {p.dtype for p in embedder._query_model.parameters() if p.is_floating_point()} == {torch.bfloat16}
    assert {p.dtype for p in embedder._index_model.parameters() if p.is_floating_point()} == {torch.float32}
    assert torch.get_default_dtype() == torch.float32


def test_searching_while_the_indexing_model_loads_still_works() -> None:
    """Loading changes PyTorch's default dtype, so searches have to wait it out safely."""
    from lodestone.embedder import GemmaEmbedder

    fresh = GemmaEmbedder()
    fresh.warm_up()
    errors: list[Exception] = []
    done = threading.Event()

    def keep_searching() -> None:
        try:
            while not done.is_set():
                assert np.isfinite(fresh.encode_query("search during a model load")).all()
        except Exception as error:
            errors.append(error)

    searcher = threading.Thread(target=keep_searching)
    searcher.start()
    try:
        vectors = fresh.encode_images([_photo(3)])
    finally:
        done.set()
        searcher.join()
    assert errors == []
    assert np.isfinite(vectors).all()


def test_search_model_unloads_when_idle_and_comes_back() -> None:
    from lodestone.embedder import GemmaEmbedder

    fresh = GemmaEmbedder(Settings(search_idle_unload_seconds=0.2))
    before = fresh.encode_query("the same search twice")
    fresh.release_search_model_if_idle()
    assert fresh.search_model_loaded  # Too soon.
    time.sleep(0.3)
    fresh.release_search_model_if_idle()
    assert not fresh.search_model_loaded
    after = fresh.encode_query("the same search twice")
    assert fresh.search_model_loaded
    assert float(np.dot(before, after)) > 0.999


def test_vectors_are_unit_length_and_finite(embedder) -> None:
    vectors = [
        embedder.encode_query("my interview notes"),
        embedder.encode_documents(["Notes from the interview about system design."])[0],
        embedder.encode_images([_photo(1)])[0],
    ]
    for vector in vectors:
        assert vector.shape == (768,)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_search_and_index_models_produce_the_same_text_vectors(embedder) -> None:
    """They use different dtypes on the CPU, which must not change what the vectors mean."""
    text = "the same sentence through both models"
    embedder.warm_up()
    query_model = embedder._query_model.encode(text, prompt_name="Document")
    index_model = embedder.encode_documents([text])[0]
    similarity = float(np.dot(query_model / np.linalg.norm(query_model), index_model))
    assert similarity > 0.999


def test_gpu_and_cpu_vectors_match_when_a_gpu_is_available() -> None:
    import torch

    from lodestone.embedder import GemmaEmbedder

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    fresh = GemmaEmbedder()
    text = "a query compared across devices"
    on_cpu = fresh.encode_query(text)
    image_cpu = fresh.encode_images([_photo(4)])[0]
    fresh.set_gpu_for_search(True)
    assert not fresh.search_model_loaded
    assert {str(p.device) for p in fresh._index_model.parameters()} == {"cpu"}  # Its own choice.
    fresh.set_gpu_for_indexing(True)
    on_gpu = fresh.encode_query(text)
    image_gpu = fresh.encode_images([_photo(4)])[0]
    assert {str(p.device) for p in fresh._query_model.parameters()} == {"cuda:0"}
    assert {p.dtype for p in fresh._query_model.parameters() if p.is_floating_point()} == {torch.bfloat16}
    assert float(np.dot(on_cpu, on_gpu)) > 0.999
    assert float(np.dot(image_cpu, image_gpu)) > 0.99
    fresh.set_gpu_for_search(False)
    assert fresh._query_model is None and fresh._index_model is not None
    fresh.set_gpu_for_indexing(False)
    assert fresh._index_model is None


def test_indexing_only_loads_the_audio_encoder_once_it_is_needed() -> None:
    from lodestone.embedder import GemmaEmbedder

    fresh = GemmaEmbedder()
    fresh.encode_images([_photo(5)])
    assert not hasattr(fresh._index_model[0].model, "audio_tower") or fresh._index_model[0].model.audio_tower is None
    before = fresh.encode_images([_photo(5)])[0]
    fresh.encode_audio(_speech(3))
    assert fresh._index_has_audio
    after = fresh.encode_images([_photo(5)])[0]
    assert float(np.dot(before, after)) > 0.999  # The larger model must not change image vectors.
    fresh.release_index_model()
    assert fresh._index_model is None and not fresh._index_has_audio


def _speech(seconds: float) -> np.ndarray:
    """Not real speech, just a changing signal that the audio encoder can tell apart from silence."""
    rng = np.random.default_rng(0)
    return (0.2 * rng.standard_normal(int(seconds * 16_000))).astype(np.float32)


def test_the_longest_audio_duration_works_and_reaches_the_end_of_the_clip(embedder) -> None:
    """By default the library cuts audio off at 30 seconds, without saying so."""
    quiet = np.zeros(MEDIA_SECONDS_MAX * 16_000, dtype=np.float32)
    ending = quiet.copy()
    ending[-int(10 * 16_000) :] = _speech(10)
    silence_vector = embedder.encode_audio(quiet)[0]
    ending_vector = embedder.encode_audio(ending)[0]
    assert np.isfinite(ending_vector).all()
    assert np.linalg.norm(ending_vector) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(silence_vector, ending_vector)) < 0.99


def test_the_most_frames_fit_even_at_the_costliest_shape(embedder) -> None:
    """Most shapes round down to fewer than 140 tokens a frame. This one uses all of them."""
    settings = Settings()
    frames = [_photo(i).resize((960, 336)) for i in range(settings.video_max_frames)]
    vector = embedder.encode_video(frames)
    assert vector.shape == (1, 768)
    assert np.isfinite(vector).all()


def test_text_finds_the_right_video_and_audio() -> None:
    """They share one space with search queries, whichever encoder made the vector."""
    from lodestone.embedder import GemmaEmbedder

    fresh = GemmaEmbedder()
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 54)

    def clip(label: str, colour: tuple[int, int, int]) -> list[Image.Image]:
        image = Image.new("RGB", (640, 360), colour)
        ImageDraw.Draw(image).text((30, 140), label, fill="white", font=font)
        return [image] * 4

    videos = {
        "beach": fresh.encode_video(clip("BEACH HOLIDAY", (30, 120, 200)))[0],
        "invoice": fresh.encode_video(clip("INVOICE TOTAL DUE", (60, 60, 60)))[0],
    }
    query = fresh.encode_query("beach holiday")
    assert float(np.dot(query, videos["beach"])) > float(np.dot(query, videos["invoice"]))


def test_inspecting_finds_everything_the_default_model_can_read(embedder) -> None:
    assert embedder.inspect_model(MODEL_ID) == DEFAULT_MODEL


def test_a_text_only_model_can_be_inspected_and_used() -> None:
    from lodestone.embedder import GemmaEmbedder

    fresh = GemmaEmbedder()
    model = fresh.inspect_model("sentence-transformers/all-MiniLM-L6-v2")
    assert model.kinds == ("text",)
    assert model.dim == 384
    assert fresh.supports("image")  # Inspecting a model doesn't switch to it.

    fresh.set_model(model)
    assert not fresh.supports("image")
    query = fresh.encode_query("a cat sitting on a mat")
    documents = fresh.encode_documents(["A cat is resting on a rug.", "Quarterly tax filing deadlines."])
    assert query.shape == (384,)
    assert documents.shape == (2, 384)
    assert float(np.dot(query, documents[0])) > float(np.dot(query, documents[1]))
    long_text = "word " * 5000  # Past this model's own limit of 256 tokens.
    assert np.isfinite(fresh.encode_documents([long_text])).all()


def test_a_model_that_does_not_exist_gives_a_clear_error(embedder) -> None:
    with pytest.raises(ValueError, match="Couldn't load"):
        embedder.inspect_model("lodestone-tests/no-such-model")


def test_ocr_reads_printed_text_and_ignores_blank_images() -> None:
    from lodestone.ocr import Ocr

    image = Image.new("RGB", (1000, 240), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
    ImageDraw.Draw(image).text((40, 90), "Invoice number 48213", fill="black", font=font)
    ocr = Ocr()
    assert "48213" in ocr.read(image).text
    assert "48213" in ocr.read_page(image).text
    assert ocr.read(Image.new("RGB", (600, 400), (200, 200, 200))).text == ""
    ocr.release()
    assert "48213" in ocr.read(image).text  # Reloads when next needed.
