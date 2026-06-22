"""Checks against the real model and OCR. Slow and needs downloads, so run with: pytest -m models"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = pytest.mark.models


def _photo(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).resize((800, 600))


@pytest.fixture(scope="module")
def embedder():
    from lodestone.embedder import GemmaEmbedder

    return GemmaEmbedder()


def test_models_loading_at_the_same_time_all_keep_one_dtype(embedder) -> None:
    """The app's startup loads both models in parallel, which once broke every image."""
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

    expected = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for model in (embedder._query_model, embedder._index_model):
        kinds = {p.dtype for p in model.parameters() if p.is_floating_point()}
        assert kinds == {expected}
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
    text = "the same sentence through both models"
    embedder.encode_images([_photo(2)])  # Make sure the indexing model is loaded.
    query_model = embedder._query_model.encode(text, prompt_name="Document")
    index_model = embedder.encode_documents([text])[0]
    similarity = float(np.dot(query_model / np.linalg.norm(query_model), index_model))
    assert similarity > 0.999


def test_ocr_reads_printed_text_and_ignores_blank_images() -> None:
    from lodestone.ocr import Ocr

    image = Image.new("RGB", (1000, 240), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
    ImageDraw.Draw(image).text((40, 90), "Invoice number 48213", fill="black", font=font)
    ocr = Ocr()
    assert "48213" in ocr.read(image).text
    assert "48213" in ocr.read_page(image).text
    assert ocr.read(Image.new("RGB", (600, 400), (200, 200, 200))).text == ""
