"""LanceDB vector storage: one table per kind of content, per embedding model."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import lancedb
import numpy as np
import pyarrow as pa

from .config import EMBEDDING_DIM, MODEL_ID, SETTINGS, Settings

log = logging.getLogger(__name__)
os.environ.setdefault("LANCE_INCLUDE_VECTOR_CENTROIDS", "false")

TEXT = "text"
IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"
VECTOR_KINDS = (TEXT, IMAGE, AUDIO, VIDEO)


@dataclass(frozen=True)
class VectorHit:
    row_id: int
    file_id: int
    similarity: float


def _table_name(kind: str, model_id: str) -> str:
    # Vectors from different models are not comparable, so they must never share a table.
    return f"{kind}_{re.sub(r'[^a-z0-9]+', '_', model_id.lower()).strip('_')}"


class VectorStore:
    def __init__(
        self,
        path: Path | str,
        model_id: str = MODEL_ID,
        dim: int = EMBEDDING_DIM,
        settings: Settings = SETTINGS,
    ) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        # Everything shares one process, but checking every read also keeps searches current
        # while the indexer is writing.
        self._db = lancedb.connect(str(path), read_consistency_interval=timedelta(0))
        self._use_model(model_id, dim)

    def _use_model(self, model_id: str, dim: int) -> None:
        self._dim = dim
        self._schema = pa.schema(
            [
                ("row_id", pa.int64()),
                ("file_id", pa.int64()),
                ("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        self._names = {kind: _table_name(kind, model_id) for kind in VECTOR_KINDS}
        self._tables = {kind: self._open(name) for kind, name in self._names.items()}

    def _open(self, name: str):
        return self._db.create_table(name, schema=self._schema, exist_ok=True)

    def switch_model(self, model_id: str, dim: int) -> None:
        """Start empty for a new model. Old tables would go stale, because reindexing changes ids."""
        with self._lock:
            self._db.drop_all_tables()
            self._use_model(model_id, dim)

    def clear(self) -> None:
        """Drop the tables rather than deleting rows, so the disk space comes back right away."""
        with self._lock:
            self._db.drop_all_tables()
            self._tables = {kind: self._open(name) for kind, name in self._names.items()}

    def count(self, kind: str) -> int:
        return self._tables[kind].count_rows()

    def add(self, kind: str, row_ids: Sequence[int], file_ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(row_ids) == 0:
            return
        values = np.asarray(vectors, dtype=np.float32)
        table = pa.table(
            {
                "row_id": pa.array(row_ids, pa.int64()),
                "file_id": pa.array(file_ids, pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(values.ravel(), pa.float32()), self._dim
                ),
            },
            schema=self._schema,
        )
        with self._lock:
            self._tables[kind].add(table)

    def delete_files(self, file_ids: Sequence[int]) -> None:
        if not file_ids:
            return
        with self._lock:
            for start in range(0, len(file_ids), 1000):
                part = ",".join(str(int(i)) for i in file_ids[start : start + 1000])
                for table in self._tables.values():
                    table.delete(f"file_id IN ({part})")

    def delete_rows(self, kind: str, row_ids: Sequence[int]) -> None:
        with self._lock:
            for start in range(0, len(row_ids), 1000):
                part = ",".join(str(int(i)) for i in row_ids[start : start + 1000])
                self._tables[kind].delete(f"row_id IN ({part})")

    def search(self, kind: str, vector: np.ndarray, limit: int) -> list[VectorHit]:
        table = self._tables[kind]
        query = (
            table.search(np.asarray(vector, dtype=np.float32), vector_column_name="vector")
            .distance_type("cosine")
            .limit(limit)
            .select(["row_id", "file_id", "_distance"])
        )
        if self.has_index(kind):
            # Search depth trades speed for accuracy. LanceDB suggests 1.5x to 10x the limit.
            query = query.ef(limit * 8)
        return [
            VectorHit(int(r["row_id"]), int(r["file_id"]), 1.0 - float(r["_distance"]))
            for r in query.to_list()
        ]

    def has_index(self, kind: str) -> bool:
        return bool(self._tables[kind].list_indices())

    def maintain(self) -> None:
        """Run after an indexing batch, while nothing else is writing.

        Compaction and cleanup are manual in LanceDB OSS. Large tables also get their vector
        index here; smaller ones stay on exact search, which is both accurate and fast enough.
        """
        with self._lock:
            for kind, table in self._tables.items():
                if not self.has_index(kind) and table.count_rows() >= self._settings.index_min_vectors:
                    log.info("building vector index for %s", kind)
                    table.create_index(
                        metric="cosine", vector_column_name="vector", index_type="IVF_HNSW_SQ"
                    )
                # Also folds new rows into the index, and removes old versions and deleted rows.
                table.optimize(cleanup_older_than=timedelta(hours=1))
