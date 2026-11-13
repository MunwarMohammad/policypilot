import os

from policypilot.config import Settings
from policypilot.embeddings import Embedder
from policypilot.store import LocalStore


def retrieval(settings: Settings):
    embedder = Embedder(settings.embedding_model)
    if settings.backend == "local":
        reranker = None
        if os.getenv("PP_RERANK") == "1":
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return LocalStore(
            settings.data_dir / "index.sqlite",
            embedder,
            hybrid=os.getenv("PP_HYBRID") == "1",
            reranker=reranker,
        )
    from policypilot.databricks_backend import DatabricksStore

    return DatabricksStore(embedder)
