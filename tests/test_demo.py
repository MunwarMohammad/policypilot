import pytest

from policypilot.demo import CORPUS, REFUSAL, build_graph, chunk_text, search, valid_evidence


def test_answer_has_source_and_trace():
    result = build_graph().invoke({"question": "supply chain", "ticker": "DEMO"})
    assert result["evidence_valid"]
    assert "[demo-1]" in result["answer"]
    assert result["trace"] == ["plan", "retrieve", "quote", "verify"]


def test_unknown_topic_refuses_before_answer():
    result = build_graph().invoke({"question": "Martian population", "ticker": "DEMO"})
    assert result["answer"] == REFUSAL
    assert result["trace"] == ["plan", "retrieve", "refuse"]


def test_company_filter_is_preserved():
    assert [c.id for c in search("supply chain", "OTHER", CORPUS)] == ["other-1"]
    assert search("supply chain", "UNKNOWN", CORPUS) == []


@pytest.mark.parametrize(
    "claim",
    [
        {"source_id": "invented", "quote": CORPUS[0].text},
        {"source_id": "demo-1", "quote": "The company has no supply chain risk."},
        {"source_id": "demo-1", "quote": ""},
    ],
)
def test_invalid_citations_or_quotes_rejected(claim):
    assert not valid_evidence([claim], list(CORPUS))


def test_cannot_cite_unretrieved_source():
    assert not valid_evidence([{"source_id": "other-1", "quote": CORPUS[2].text}], [CORPUS[0]])


def test_chunk_overlap_and_tail():
    assert chunk_text("a b c d e f", size=4, overlap=1) == ["a b c d", "d e f"]
    assert chunk_text("") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, 3), (3, -1)])
def test_bad_chunk_configuration_fails(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("example", size, overlap)
