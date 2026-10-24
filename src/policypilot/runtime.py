from policypilot.config import Settings
from policypilot.embeddings import Embedder
from policypilot.store import LocalStore


def retrieval(settings: Settings):
    embedder = Embedder(settings.embedding_model)
    if settings.backend == "local":
        return LocalStore(settings.data_dir / "index.sqlite", embedder)
    from policypilot.databricks_backend import DatabricksStore

    return DatabricksStore(embedder)
