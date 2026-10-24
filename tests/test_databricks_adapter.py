import json
from types import SimpleNamespace

from test_pipeline import FakeEmbedder, passage

from policypilot.databricks_backend import DatabricksStore


def test_rest_adapter_maps_manifest_order_and_forwards_filters():
    p = passage()
    columns = ["embedding_model", "score", *reversed(list(type(p).model_fields))]
    values = {**p.model_dump(), "embedding_model": "test-v1", "score": 0.9}
    calls = []

    def request(method, path, body):
        calls.append((method, path, body))
        return {
            "manifest": {"columns": [{"name": k} for k in columns]},
            "result": {"data_array": [[values[k] for k in columns]]},
        }

    store = object.__new__(DatabricksStore)
    store.embedder = FakeEmbedder()
    store.index_name = "dev.filings.index"
    store.client = SimpleNamespace(api_client=SimpleNamespace(do=request))
    result = store.search("supply", "AAPL")
    assert result[0].passage == p
    assert result[0].score == 0.9
    assert json.loads(calls[0][2]["filters_json"]) == {
        "ticker": "AAPL",
        "embedding_model": "test-v1",
    }
