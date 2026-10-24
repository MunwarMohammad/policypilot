"""Small-corpus local vector search: atomic SQLite writes and cosine ranking.

Exhaustive search keeps the prototype transparent; production uses a managed index.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from policypilot.models import Hit, Passage


class LocalStore:
    def __init__(self, path: Path, embedder):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.embedder = path, embedder
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, ticker TEXT, "
                "report_date TEXT, payload TEXT, vector TEXT)"
            )
            saved = db.execute("SELECT value FROM config WHERE key='model'").fetchone()
            if saved and saved[0] != embedder.model_name:
                raise ValueError("Embedding model changed: use a new PP_DATA_DIR and re-ingest")
            db.execute("INSERT OR IGNORE INTO config VALUES ('model', ?)", (embedder.model_name,))

    def connect(self):
        return sqlite3.connect(self.path)

    def replace_company(self, ticker: str, passages: list[Passage]):
        if not passages or any(p.ticker != ticker for p in passages):
            raise ValueError("Expected nonempty passages for exactly one company")
        vectors = np.asarray(self.embedder.encode([p.text for p in passages]), dtype=float)
        if vectors.ndim != 2 or len(vectors) != len(passages) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding output")
        with self.connect() as db:
            # A failed insert rolls back the deletion; readers never see partial ingestion.
            db.execute("DELETE FROM chunks WHERE ticker=?", (ticker,))
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        p.chunk_id,
                        p.ticker,
                        p.report_date,
                        p.model_dump_json(),
                        json.dumps(v.tolist()),
                    )
                    for p, v in zip(passages, vectors, strict=True)
                ],
            )

    def search(self, query: str, ticker: str, k: int = 5) -> list[Hit]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload, vector FROM chunks WHERE ticker=?", (ticker,)
            ).fetchall()
        if not rows:
            return []
        vectors = np.asarray([json.loads(row[1]) for row in rows], dtype=float)
        q = np.asarray(self.embedder.encode([query])[0], dtype=float)
        if vectors.shape[1] != q.shape[0]:
            raise ValueError("Embedding dimensions changed; rebuild the index")
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
        scores = np.divide(vectors @ q, denom, out=np.zeros(len(rows)), where=denom > 0)
        indices = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(passage=Passage.model_validate_json(rows[i][0]), score=float(scores[i]))
            for i in indices
        ]

    def export(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload, vector FROM chunks ORDER BY id").fetchall()
        return [
            {
                **json.loads(p),
                "embedding": json.loads(v),
                "embedding_model": self.embedder.model_name,
            }
            for p, v in rows
        ]
