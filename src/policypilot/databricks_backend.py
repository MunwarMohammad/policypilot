"""Adapter for an existing STANDARD Delta Sync Vector Search index."""

import json
import os
from urllib.parse import quote

from policypilot.models import Hit, Passage

COLUMNS = list(Passage.model_fields)


class DatabricksStore:
    def __init__(self, embedder):
        # SDK unified auth supports locally configured profiles and app service principals.
        from databricks.sdk import WorkspaceClient

        self.client = WorkspaceClient()
        self.embedder = embedder
        self.index_name = os.environ["PP_VECTOR_INDEX"]

    def search(self, query: str, ticker: str, k: int = 5):
        # Using the workspace REST API keeps SDK OAuth refresh intact in long-lived apps.
        raw = self.client.api_client.do(
            "POST",
            "/api/2.0/vector-search/indexes/" + quote(self.index_name, safe="") + "/query",
            body={
                "query_vector": self.embedder.encode([query])[0].tolist(),
                "columns": COLUMNS + ["embedding_model"],
                "num_results": k,
                "filters_json": json.dumps(
                    {"ticker": ticker, "embedding_model": self.embedder.model_name}
                ),
            },
        )
        names = [c["name"] for c in raw["manifest"]["columns"]]
        hits = []
        for values in raw.get("result", {}).get("data_array", []):
            row = dict(zip(names, values, strict=True))
            if row.get("embedding_model") != self.embedder.model_name:
                raise ValueError("Index and query embedding models differ")
            hits.append(
                Hit(passage=Passage(**{k: row[k] for k in COLUMNS}), score=float(row["score"]))
            )
        return hits
