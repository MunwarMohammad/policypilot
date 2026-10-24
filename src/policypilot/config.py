"""Configuration is loaded at runtime so imports never need credentials."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

TICKERS = (
    "AAPL",
    "MSFT",
    "JPM",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JNJ",
    "V",
    "WMT",
    "CVX",
    "PG",
    "HD",
    "BAC",
    "KO",
    "PEP",
    "CSCO",
    "INTC",
    "DIS",
)


@dataclass(frozen=True)
class Settings:
    backend: str
    data_dir: Path
    embedding_model: str
    min_score: float

    @classmethod
    def load(cls):
        load_dotenv()
        backend = os.getenv("PP_BACKEND", "local")
        if backend not in {"local", "databricks"}:
            raise ValueError("PP_BACKEND must be local or databricks")
        score = float(os.getenv("PP_MIN_SCORE", "0.30"))
        if not -1 <= score <= 1:
            raise ValueError("PP_MIN_SCORE must be between -1 and 1")
        return cls(
            backend,
            Path(os.getenv("PP_DATA_DIR", "data")),
            os.getenv("PP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            score,
        )
