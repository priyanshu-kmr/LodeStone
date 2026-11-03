"""Query cleanup, the three retrieval lists, and merging them by rank."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from .catalog import Catalog, FileRow, query_terms, term_coverage
from .config import SETTINGS, Settings, is_under
from .vectors import TEXT, VECTOR_KINDS, VectorHit, VectorStore

ALL = "all"
IMAGES = "images"
DOCUMENTS = "documents"
VIDEOS = "videos"
AUDIO = "audio"
# The kind of file each of these filters keeps. Documents are whatever is left over.
_MODE_KINDS = {IMAGES: "image", VIDEOS: "video", AUDIO: "audio"}

_LEAD = r"^(?:please\s+)?(?:(?:find|show|search(?:\s+for)?|get|open|look\s+for)\s+(?:me\s+)?)?(?:(?:the|an?|my)\s+)?"
_LINK = (
    r"(?:\s+(?:of|about|on|with|showing|containing|having|called|named|where"
    r"|(?:which|that)\s+(?:has|have|had|shows?|contains?|includes?|is|was|mentions?)))?\b\s*"
)
_IMAGE_QUERY = re.compile(
    _LEAD + r"(?:images?|photos?|pictures?|pics?|screenshots?)\b" + _LINK, re.IGNORECASE
)
_DOC_QUERY = re.compile(_LEAD + r"(?:documents?|docs?|pdfs?)\b" + _LINK, re.IGNORECASE)
_VIDEO_QUERY = re.compile(_LEAD + r"(?:videos?|movies?)\b" + _LINK, re.IGNORECASE)
_AUDIO_QUERY = re.compile(_LEAD + r"(?:audio|recordings?|songs?|sounds?)\b" + _LINK, re.IGNORECASE)
_FILE_QUERY = re.compile(_LEAD + r"files?\b" + _LINK, re.IGNORECASE)


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class ParsedQuery:
    text: str
    mode: str = ALL


@dataclass
class SearchResult:
    file_id: int
    path: str
    name: str
    kind: str
    score: float
    snippet: str = ""
    matched_by: list[str] = field(default_factory=list)
    index_level: str = "full"


def parse_query(query: str, mode: str = ALL) -> ParsedQuery:
    """Strip filler like 'file which contains', and treat 'image of' as an image filter."""
    query = " ".join(query.split())
    for pattern, implied in (
        (_IMAGE_QUERY, IMAGES),
        (_DOC_QUERY, DOCUMENTS),
        (_VIDEO_QUERY, VIDEOS),
        (_AUDIO_QUERY, AUDIO),
        (_FILE_QUERY, ALL),
    ):
        match = pattern.match(query)
        if match:
            rest = query[match.end() :].strip()
            # A query that is only 'photo' has nothing left to search for, so keep it as typed.
            if rest:
                return ParsedQuery(rest, mode if mode != ALL else implied)
            break
    return ParsedQuery(query, mode)


def standout_scores(similarities: dict[int, float], min_spread: float) -> dict[int, float]:
    """How far each hit stands out from the rest of its list, in standard deviations.

    Raw similarities from text and image vectors are not comparable. A real match clearly
    stands out from its own list, though, and that can be compared across lists.
    """
    if not similarities:
        return {}
    values = np.fromiter(similarities.values(), dtype=np.float64)
    spread = max(float(values.std()), min_spread)
    middle = float(values.mean())
    return {file_id: (value - middle) / spread for file_id, value in similarities.items()}


def rrf_merge(
    lists: dict[str, list[int]],
    k: int,
    weights: dict[str, dict[int, float]] | None = None,
) -> tuple[dict[int, float], dict[int, list[str]]]:
    """Reciprocal rank fusion, with an optional weight per hit."""
    scores: dict[int, float] = {}
    matched: dict[int, list[str]] = {}
    for label, ids in lists.items():
        hit_weights = (weights or {}).get(label, {})
        for rank, file_id in enumerate(ids, start=1):
            scores[file_id] = scores.get(file_id, 0.0) + hit_weights.get(file_id, 1.0) / (k + rank)
            matched.setdefault(file_id, []).append(label)
    return scores, matched


def _best_per_file(hits: Iterable[VectorHit]) -> dict[int, float]:
    best: dict[int, float] = {}
    for hit in hits:
        best[hit.file_id] = max(best.get(hit.file_id, -1.0), hit.similarity)
    return best


class Searcher:
    def __init__(
        self,
        catalog: Catalog,
        vectors: VectorStore,
        encoder: QueryEncoder,
        settings: Settings = SETTINGS,
    ) -> None:
        self._catalog = catalog
        self._vectors = vectors
        self._encoder = encoder
        self._settings = settings

    def search(self, query: str, top_k: int | None = None, mode: str = ALL) -> list[SearchResult]:
        parsed = parse_query(query, mode)
        if not parsed.text:
            return []
        top_k = top_k or self._settings.default_top_k
        limit = max(self._settings.candidates_per_list, top_k * 3)
        snippet_chunks: dict[int, int] = {}

        vector = self._encoder.encode_query(parsed.text)
        hits = {kind: self._vectors.search(kind, vector, limit) for kind in VECTOR_KINDS}
        for hit in hits[TEXT]:  # Best first, so each file keeps its most relevant passage.
            snippet_chunks.setdefault(hit.file_id, hit.row_id)

        # One 'meaning' ranking: for each file, whichever of its vectors stands out the most.
        spread = self._settings.min_score_spread
        meaning: dict[int, float] = {}
        for kind_hits in hits.values():
            for file_id, score in standout_scores(_best_per_file(kind_hits), spread).items():
                meaning[file_id] = max(meaning.get(file_id, -np.inf), score)
        lists = {"meaning": sorted(meaning, key=lambda i: meaning[i], reverse=True)}

        keyword_files, keyword_weights = self._keyword_list(parsed.text, limit, snippet_chunks)
        lists["keyword"] = keyword_files

        scores, matched = rrf_merge(lists, self._settings.rrf_k, {"keyword": keyword_weights})
        files = self._catalog.get_files_by_ids(scores)
        inactive = [f.path for f in self._catalog.folders() if not f.enabled]
        # Equal fusion scores are common at the top, so let the stronger meaning match decide.
        ranked = sorted(
            scores, key=lambda i: (scores[i], meaning.get(i, -np.inf)), reverse=True
        )

        chosen: list[FileRow] = []
        for file_id in ranked:
            row = files.get(file_id)
            if row is None or not self._keep(row, parsed.mode, inactive):
                continue
            # Below its own list's average, with no keyword match, it is only padding.
            if file_id not in keyword_weights and meaning.get(file_id, -np.inf) < 0:
                continue
            chosen.append(row)
            if len(chosen) == top_k:
                break

        texts = self._catalog.chunk_texts(
            snippet_chunks[r.id] for r in chosen if r.id in snippet_chunks
        )
        results = []
        for row in chosen:
            source_text = texts.get(snippet_chunks.get(row.id, -1))
            snippet = ""
            if source_text and source_text[0] != "name":
                snippet = " ".join(source_text[1].split())[:240]
            results.append(
                SearchResult(
                    row.id, row.path, row.name, row.kind, scores[row.id], snippet,
                    matched[row.id], row.index_level,
                )
            )
        return results

    def _keyword_list(
        self, text: str, limit: int, snippet_chunks: dict[int, int]
    ) -> tuple[list[int], dict[int, float]]:
        """Keyword hits, weighted by how much of the query they cover.

        Matching every word counts double, because exact matches are what keywords are for.
        Matching under half is dropped, or one common word would outrank a real match by meaning.
        """
        hits = self._catalog.keyword_search(text, limit)
        if not hits:
            return [], {}
        terms = query_terms(text)
        chunks = self._catalog.chunk_texts(chunk_id for chunk_id, _ in hits)
        weights: dict[int, float] = {}
        ordered: list[tuple[int, int, bool]] = []
        for chunk_id, file_id in hits:
            source, chunk = chunks.get(chunk_id, ("", ""))
            coverage = term_coverage(terms, chunk)
            if coverage < self._settings.keyword_min_coverage:
                continue
            complete = coverage >= 1.0
            ordered.append((file_id, chunk_id, complete))
            weights[file_id] = max(weights.get(file_id, 0.0), 2.0 if complete else 1.0)
            # A passage that matched exactly beats one that matched by meaning, but a filename
            # match has no passage worth showing.
            if source != "name" and (complete or file_id not in snippet_chunks):
                snippet_chunks[file_id] = chunk_id
        # Complete matches first, keeping BM25's order within each group.
        ordered.sort(key=lambda hit: not hit[2])
        return list(dict.fromkeys(file_id for file_id, _, _ in ordered)), weights

    @staticmethod
    def _keep(row: FileRow, mode: str, inactive: list[str]) -> bool:
        if mode in _MODE_KINDS and row.kind != _MODE_KINDS[mode]:
            return False
        if mode == DOCUMENTS and row.kind in _MODE_KINDS.values():
            return False
        return not any(is_under(row.path, folder) for folder in inactive)
