# LodeStone

Local semantic search for files and images on Windows. 

## Run it

Double-click `LodeStone.bat`. The first run downloads the embedding model (about 1.5 GB) and the
OCR models.

Closing the window keeps it running in the system tray, so it can keep indexing and watching for
new files. Use **Quit** in the tray menu to stop it.

It indexes Desktop, Documents, Downloads and Pictures by default. Change this under
**Settings > Folders**. Your files are only ever read, never changed, moved or deleted.

The index is stored in `%LOCALAPPDATA%\LodeStone`. Set `LODESTONE_DATA` to use another location.
It holds copies of your documents' text, so treat it as being as sensitive as those files.

### Command line

```
.venv\Scripts\lodestone index --folder "C:\Some\Folder"
.venv\Scripts\lodestone search "blood test results" --top 5
.venv\Scripts\lodestone status
```

### Set up from scratch

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

## How it works

```
INDEXING (background)
  watcher events + periodic rescans -> queue in SQLite -> worker checks the file's real state
    text / PDF / Office -> extract text -> chunks -> embed text --\
    scanned PDF pages   -> OCR ---------> chunks -> embed text ----> LanceDB (vectors)
    images              -> quality gate -> OCR + embed image -----/  SQLite (files, text, FTS5)

SEARCH (text encoder only)
  query -> clean up -> embed -> text vectors + image vectors -> one "meaning" ranking --\
                    -> FTS5 keyword search, weighted by how many query words matched ----> RRF
```

| Part | Choice | Why |
|---|---|---|
| Embeddings | EmbeddingGemma 2 (`google/embeddinggemma-2`) | One Apache 2.0 model puts text and images in the same 768-dimension space |
| OCR | RapidOCR + PP-OCRv6 small, on the CPU | No PaddlePaddle dependency, and it leaves the GPU free |
| Vectors | LanceDB, one table per model and type | Runs inside the app, with indexes on disk rather than in RAM |
| Files, text, keywords | SQLite + FTS5 | Source of truth, with transactional updates and BM25 |
| Vector index | Exact search under 10,000 vectors, then `IVF_HNSW_SQ` | Exact search is accurate and fast enough until then |
| Change detection | `watchdog` plus a rescan at startup and every 6 hours | Events can be lost, so they are only hints |
| UI | PySide6 | Same language as the backend, with a native folder picker |

### Decisions worth explaining

**Ranking.** The first version used plain reciprocal rank fusion (RRF) across the text, image and
keyword lists. On a test set it put the right file first for only 15 of 26 queries, even though the
correct file was first in its own list. Three things were wrong:

1. OCR reads stray marks like `O` or `8` in ordinary photos, and very short text gets a generic
   vector that scores highly against everything. Those marks are now filtered out, and very short
   text only goes into keyword search.
2. RRF treats rank 1 of every list as equally trustworthy, even when nothing in that list is
   relevant. Raw text and image similarities aren't comparable either. Each hit is now scored by
   how far it stands out from its own list, in standard deviations, and text and image hits merge
   into a single meaning ranking.
3. A keyword match on one word of a long query got too much credit. Keyword hits are now weighted
   by how many of the query's words they cover. RRF then merges meaning and keywords.

**Crash safety.** A file is only marked `indexed` after its vectors are written. Its queue entry is
only removed after that. Anything unfinished gets redone on restart, and old vectors are deleted
first, so nothing is duplicated.

**Moved files.** If a file's hash matches a known file whose old path is gone, it moved. The path
is updated and the vectors are kept, because image embeddings are expensive.

**Low-quality images.** Images under 200x200 on both sides, or that are blank, very blurry or
unreadable, skip OCR and embedding but stay findable by filename. **Settings > Name-only images**
has an **Index anyway** button for mistakes. Wrongly hiding an image is worse than indexing a bad
one, so the blur threshold is deliberately low.

**Datasets.** A folder with 1,000+ images and at least two other signs (numbered names, very small
files, nearby code, dataset-style names) pauses and asks before indexing. It asks rather than skips
because silently hiding someone's files would be worse.

## Measured results

Measured on a laptop with an RTX 3060 (6 GB), a Ryzen 7 5800H and 15 GB of RAM.

| What | Result |
|---|---|
| Search, while idle | About 100 ms (a little over 1,000 files) |
| Search, while indexing | About 260 ms typically, 510 ms at the 95th percentile, one 8 s outlier |
| VRAM for search | 0.5 GB, plus up to 2 GB while indexing images |
| Image embedding | About 5 images per second |
| OCR | About 0.5 s for a photo, about 1.1 s for a scanned page |
| Index size | 94 MB for 1,015 files |
| `IVF_HNSW_SQ` recall | 0.955 for text, 0.970 for images, at 20 results (forced on at 500 vectors) |

Ranking was checked with public test photos and generated documents, mixed with about 1,000
real files as distractors. These query sets are small, so treat them as a rough guide:

| Query set | Right file first | In the top 3 | In the top 10 |
|---|---|---|---|
| 26 queries used while tuning | 21 | 24 | 26 |
| 20 new queries, not used for tuning | 11 | 17 | 20 |

Some "wrong" first results may be reasonable matches among the real files. Others are clear
errors, such as a checklist ranking above a retina photo for "anatomy of the eye". Choosing the
**Images** or **Documents** filter helps with those, and so does starting a query with "image of".

The tests: `pytest` runs the fast suite with a fake embedder, and `pytest -m models --override-ini=addopts=`
runs the checks against the real models.

## Known limits

- Very blurry or dark images can be misjudged by the quality gate, and handwriting is unreliable
  in OCR.
- Search always returns its closest matches, so a query with no real match can still show weak
  results.
- Indexing uses about 5 to 6 CPU cores during OCR, so the laptop can feel busy on the first run.
  **Settings > Pause indexing** stops it at any time, and it resumes where it left off.
- Only about the first 500 tokens of each file are embedded (estimated at 4 characters per token,
  and adjustable in `config.py`). Meaning search can't find something that only appears later in a
  file. Text-layer documents stay keyword-searchable up to 300 chunks, and spreadsheets read the first
  1,000 rows of each sheet.
- OCR on a scanned PDF stops once the file has about 500 tokens of text, usually after a page or
  two, so later scanned pages aren't searchable at all, even by keyword.
- OCR runs on the CPU, and the GPU only does embeddings, so the GPU sits idle for much of a first
  run.
- Office and HTML files over 100 MB are name-only, and any conversion that takes over 2 minutes is
  stopped, so one file can't block the rest.
- `.doc`, `.ppt`, `.xls` and `.rtf` are only searchable by filename. HEIC photos need extra
  software that isn't included, so they are also name-only.
- Code files aren't indexed, because they would crowd out everything else.
- Windows only. PyMuPDF is AGPL, which matters if this is ever distributed as a closed-source
  product.
