# LodeStone

Local semantic search for files, images, audio and video on Windows.

## Setup

Run these in the root of this directory:

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

Then double-click `LodeStone.bat`. The first run downloads the models (about 1.5 GB), then indexes
Desktop, Documents, Downloads and Pictures.

Closing the window keeps it indexing in the system tray. Use **Quit** in the tray menu to stop it.

The index is in `%LOCALAPPDATA%\LodeStone` and holds copies of your documents' text.

## How it works

![LodeStone architecture](docs/architecture.png)

| Part | Method |
|---|---|
| Embeddings | EmbeddingGemma 2 (text, images, audio and video in one vector space) |
| OCR | RapidOCR with PP-OCRv6, on the CPU |
| Audio and video decoding | PyAV |
| Vectors | LanceDB, exact search under 10,000 vectors|
| Files, text, keywords | SQLite with FTS5 |
| File changes | `watchdog`, plus a rescan at startup and every 6 hours |
| UI | PySide6 |

## Files

In `src/lodestone/`:

| File | What it does |
|---|---|
| `engine.py` | Runs the background work and connects everything: indexing, rescans, search and settings changes |
| `config.py` | Paths, supported file types and tuning values |
| `scanner.py` | Walks folders, applies the folder rules and spots image datasets |
| `watcher.py` | Watches for file changes and queues those paths to be checked |
| `indexer.py` | Works through the queue and writes each file to the catalog and vector store |
| `extract.py` | Turns documents into plain text and splits it into chunks |
| `office.py` | Converts Office files in a separate process, so one bad file can't block indexing |
| `ocr.py` | Reads text from images and scanned pages |
| `quality.py` | Decides which images are good enough to embed |
| `media.py` | Decodes the opening of audio and video files |
| `embedder.py` | Loads and unloads the embedding model, and turns content into vectors |
| `vectors.py` | Stores and searches vectors in LanceDB |
| `catalog.py` | SQLite database of files, text, keyword search, the queue and settings |
| `search.py` | Cleans up the query, then merges the meaning and keyword results into one ranking |
| `cli.py` | Command line: `lodestone index`, `lodestone search "..."` and `lodestone status` |
| `ui/app.py` | The search window, settings dialog and system tray |

## License

[AGPL-3.0](LICENSE)
