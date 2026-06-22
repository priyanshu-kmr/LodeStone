"""Command line: `lodestone index`, `lodestone search "..."`, `lodestone status`."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .engine import Engine
from .search import ALL, DOCUMENTS, IMAGES


def _print_status(engine: Engine) -> None:
    status = engine.status()
    print(
        f"{status.state}: {status.indexed} indexed, {status.queued} queued, "
        f"{status.name_only} images by name only, {status.errors} errors"
    )
    for prompt in engine.dataset_prompts():
        print(f"  needs a decision: {prompt.path} ({prompt.image_count} images; {prompt.signs})")


def _index(args: argparse.Namespace) -> int:
    engine = Engine(watch=False)
    if args.folder:
        for folder in args.folder:
            engine.add_folder(folder)
    else:
        engine.seed_defaults()
    started = time.perf_counter()
    engine.start()
    try:
        engine.wait_until_idle(timeout=args.timeout)
    finally:
        engine.stop()
    print(f"Finished in {time.perf_counter() - started:.1f}s")
    _print_status(engine)
    return 0


def _search(args: argparse.Namespace) -> int:
    engine = Engine(watch=False)
    started = time.perf_counter()
    results = engine.search(args.query, args.top, args.mode)
    elapsed = (time.perf_counter() - started) * 1000
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>2}. {result.name}  [{'+'.join(result.matched_by)}]")
        print(f"    {result.path}")
        if result.snippet:
            print(f"    {result.snippet[:160]}")
    print(f"{len(results)} results in {elapsed:.0f} ms (includes loading the model)")
    return 0


def _status(_: argparse.Namespace) -> int:
    from .config import get_paths

    print(f"Index folder: {get_paths().root}")
    _print_status(Engine(embedder=object(), ocr=object(), watch=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="lodestone", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="index folders, then exit")
    index.add_argument("--folder", action="append", help="add this folder (repeatable)")
    index.add_argument("--timeout", type=float, default=24 * 3600)
    index.set_defaults(run=_index)

    search = commands.add_parser("search", help="search the index")
    search.add_argument("query")
    search.add_argument("--top", type=int, default=10)
    search.add_argument("--mode", choices=[ALL, IMAGES, DOCUMENTS], default=ALL)
    search.set_defaults(run=_search)

    status = commands.add_parser("status", help="show index status")
    status.set_defaults(run=_status)

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
