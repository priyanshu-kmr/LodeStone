"""The separate conversion process."""

from __future__ import annotations

import pytest

from lodestone.office import OfficeWorker, convert_office_file
from office_helpers import fail_on_bad_names, hang_forever


def test_timeout_kills_the_process_and_the_next_file_still_works(tmp_path) -> None:
    stuck = OfficeWorker(timeout=1.5, handler=hang_forever)
    with pytest.raises(TimeoutError):
        stuck.convert("anything")
    stuck.close()

    path = tmp_path / "page.html"
    path.write_text("<html><body><p>hello from the worker</p></body></html>")
    worker = OfficeWorker(timeout=120, handler=convert_office_file)
    try:
        assert "hello from the worker" in worker.convert(str(path))
    finally:
        worker.close()


def test_a_bad_file_reports_an_error_and_the_process_survives() -> None:
    worker = OfficeWorker(timeout=120, handler=fail_on_bad_names)
    try:
        with pytest.raises(ValueError, match="can't be converted"):
            worker.convert("bad.pptx")
        assert worker.convert("good.pptx") == "converted"
    finally:
        worker.close()
