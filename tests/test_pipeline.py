import json

import numpy as np
import pytest

from policypilot.agent import REFUSAL, build_agent, check_citations
from policypilot.models import Answer, Claim, Hit, Judgment, Passage, Plan
from policypilot.sec import SECClient, clean_html
from policypilot.store import LocalStore


def passage(
    ticker="AAPL", text="Supply disruptions could delay product deliveries.", identifier="c1"
):
    return Passage(
        ticker=ticker,
        cik=1,
        company="Test company",
        accession="0001-26-001",
        filing_date="2026-01-01",
        report_date="2025-12-31",
        source_url="https://www.sec.gov/test",
        chunk_id=identifier,
        chunk_index=0,
        text=text,
    )


class FakeEmbedder:
    model_name = "test-v1"

    def encode(self, texts):
        return np.array(
            [[float("supply" in t.lower()), float("cyber" in t.lower())] for t in texts]
        )


class FakeModel:
    def __init__(self, p, supported=True, bad_quote=False):
        self.p, self.supported, self.bad_quote = p, supported, bad_quote
        self.calls = []

    def generate(self, instruction, payload, schema):
        self.calls.append(schema)
        if schema == Plan:
            return Plan(query="supply")
        if schema == Answer:
            return Answer(
                claims=[
                    Claim(
                        text="Deliveries may be delayed.",
                        source_id=self.p.chunk_id,
                        quote="an invented quotation" if self.bad_quote else self.p.text,
                    )
                ]
            )
        return Judgment(supported=[self.supported])


class FakeStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, ticker, k):
        self.calls.append((query, ticker))
        return self.hits


def test_real_graph_publishes_only_verified_claims():
    p = passage()
    store = FakeStore([Hit(passage=p, score=0.9)])
    result = build_agent(FakeModel(p), store).invoke(
        {"question": "supply risks?", "ticker": "AAPL"}
    )
    assert result["answer"] == "Deliveries may be delayed. [1]"
    assert result["trace"] == ["plan", "retrieve", "draft", "verify", "answer"]
    assert store.calls == [("supply", "AAPL")]


@pytest.mark.parametrize(
    "hits", [[], [Hit(passage=passage("MSFT"), score=0.9)], [Hit(passage=passage(), score=0.1)]]
)
def test_no_evidence_retries_once_with_company_filter(hits):
    store = FakeStore(hits)
    model = FakeModel(passage())
    result = build_agent(model, store).invoke({"question": "original question", "ticker": "AAPL"})
    assert result["answer"] == REFUSAL
    assert store.calls == [("supply", "AAPL"), ("original question", "AAPL")]
    assert model.calls == [Plan]


@pytest.mark.parametrize("supported,bad_quote", [(False, False), (True, True)])
def test_unsupported_or_fabricated_answer_refuses(supported, bad_quote):
    p = passage()
    result = build_agent(
        FakeModel(p, supported, bad_quote), FakeStore([Hit(passage=p, score=1)])
    ).invoke({"question": "supply", "ticker": "AAPL"})
    assert result["answer"] == REFUSAL


def test_unknown_source_cannot_pass_with_real_quote():
    p = passage()
    answer = Answer(claims=[Claim(text="test", source_id="invented", quote=p.text)])
    assert not check_citations(answer, [Hit(passage=p, score=1)])


def test_replace_is_idempotent_and_company_scoped(tmp_path):
    store = LocalStore(tmp_path / "index.sqlite", FakeEmbedder())
    store.replace_company("AAPL", [passage()])
    store.replace_company("MSFT", [passage("MSFT", identifier="m1")])
    store.replace_company("AAPL", [passage(identifier="new")])
    assert len(store.export()) == 2
    assert store.search("supply", "AAPL")[0].passage.chunk_id == "new"
    assert store.search("supply", "MSFT")[0].passage.chunk_id == "m1"
    assert store.search("supply", "UNKNOWN") == []


def test_failed_transaction_preserves_existing_rows(tmp_path):
    import sqlite3

    store = LocalStore(tmp_path / "index.sqlite", FakeEmbedder())
    store.replace_company("AAPL", [passage()])
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_company(
            "AAPL", [passage(identifier="duplicate"), passage(identifier="duplicate")]
        )
    assert store.search("supply", "AAPL")[0].passage.chunk_id == "c1"


def test_model_change_is_rejected(tmp_path):
    LocalStore(tmp_path / "index.sqlite", FakeEmbedder())
    embedder = FakeEmbedder()
    embedder.model_name = "different"
    with pytest.raises(ValueError, match="model changed"):
        LocalStore(tmp_path / "index.sqlite", embedder)


def test_html_removes_hidden_instructions():
    assert clean_html("<script>bad</script><style>bad</style><p>Supply risk</p>") == "Supply risk"


class Response:
    def __init__(self, payload, status=200):
        self.text = json.dumps(payload)
        self.status_code, self.headers = status, {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self, responses):
        self.responses, self.headers, self.urls = iter(responses), {}, []

    def get(self, url, timeout):
        self.urls.append(url)
        return next(self.responses)


def test_sec_falls_back_to_old_submissions(tmp_path):
    session = Session(
        [
            Response({"0": {"ticker": "AAPL", "cik_str": 1}}),
            Response(
                {
                    "name": "Test",
                    "filings": {"recent": {"form": ["8-K"]}, "files": [{"name": "older.json"}]},
                }
            ),
            Response(
                {
                    "form": ["10-K"],
                    "accessionNumber": ["0001-26-001"],
                    "primaryDocument": ["annual.htm"],
                    "filingDate": ["2026-01-01"],
                    "reportDate": ["2025-12-31"],
                }
            ),
        ]
    )
    client = SECClient("PolicyPilot developer@valid.org", tmp_path, session, lambda _: None)
    filing = client.latest("AAPL")
    assert filing.report_date == "2025-12-31"
    assert filing.source_url.endswith("/1/000126001/annual.htm")
    assert session.urls[-1].endswith("older.json")


def test_sec_retries_rate_limit_and_caches_document(tmp_path):
    session = Session([Response({}, 429), Response({"ok": True})])
    client = SECClient("PolicyPilot developer@valid.org", tmp_path, session, lambda _: None)
    assert client.get("https://www.sec.gov/test", immutable=True) == '{"ok": true}'
    assert client.get("https://www.sec.gov/test", immutable=True) == '{"ok": true}'
    assert len(session.urls) == 2


def test_ingestion_reports_partial_failure(tmp_path):
    from policypilot.ingestion import ingest

    class Client:
        def latest(self, ticker):
            raise ValueError("unavailable")

    report = ingest(["AAPL"], Client(), None, None, tmp_path)
    assert report[0]["status"] == "failed"
    assert json.loads((tmp_path / "ingestion-report.json").read_text())["results"] == report
