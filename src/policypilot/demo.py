"""Lesson 1: inspectable retrieval and evidence validation with real LangGraph.

This uses fictional data, keyword retrieval and exact quotes, not an LLM or
semantic embeddings. Keeping these boundaries explicit makes the graph easy to learn.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


@dataclass(frozen=True)
class Chunk:
    id: str
    ticker: str
    text: str
    source: str


CORPUS = (
    Chunk(
        "demo-1",
        "DEMO",
        "Supply chain disruptions may delay component deliveries.",
        "Fictional training document, section 1",
    ),
    Chunk(
        "demo-2",
        "DEMO",
        "Cybersecurity incidents may interrupt business operations.",
        "Fictional training document, section 2",
    ),
    Chunk(
        "other-1",
        "OTHER",
        "Supplier concentration increases supply chain risk.",
        "Another fictional training document, section 1",
    ),
)
STOP_WORDS = {"what", "which", "does", "the", "a", "an", "of", "is", "are", "about", "tell", "me"}
REFUSAL = "I could not find sufficient evidence in the selected documents."


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Word windows for this lesson; semantic ingestion will use model token limits."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    tokens = text.split()
    result = []
    for start in range(0, len(tokens), size - overlap):
        result.append(" ".join(tokens[start : start + size]))
        if start + size >= len(tokens):
            break
    return result


def search(query: str, ticker: str, corpus: tuple[Chunk, ...], k: int = 3) -> list[Chunk]:
    """A transparent keyword baseline, deliberately not called vector search."""
    query_words = words(query) - {ticker.lower()}
    scored = [(len(query_words & words(c.text)), c) for c in corpus if c.ticker == ticker]
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [chunk for score, chunk in scored[:k] if score > 0]


def valid_evidence(claims: list[dict], sources: list[Chunk]) -> bool:
    """Validate source identity and exact quotation; this does NOT prove relevance.

    Later generated paraphrases need additional entailment evaluation. Citation
    formatting alone cannot establish that a factual claim is supported.
    """
    by_id = {c.id: c for c in sources}
    if not claims:
        return False
    for claim in claims:
        source = by_id.get(claim.get("source_id"))
        quote = claim.get("quote")
        if source is None or not isinstance(quote, str) or not quote.strip():
            return False
        if quote not in source.text:
            return False
    return True


class State(TypedDict, total=False):
    question: str
    ticker: str
    query: str
    sources: list[Chunk]
    claims: list[dict]
    answer: str
    evidence_valid: bool
    trace: list[str]


def build_graph(corpus: tuple[Chunk, ...] = CORPUS):
    def plan(state: State):
        return {"query": state["question"].strip(), "trace": ["plan"]}

    def retrieve(state: State):
        return {
            "sources": search(state["query"], state["ticker"], corpus),
            "trace": state["trace"] + ["retrieve"],
        }

    def quote(state: State):
        return {
            "claims": [{"source_id": c.id, "quote": c.text} for c in state["sources"]],
            "trace": state["trace"] + ["quote"],
        }

    def verify(state: State):
        valid = valid_evidence(state["claims"], state["sources"])
        answer = (
            "\n".join(f'"{c["quote"]}" [{c["source_id"]}]' for c in state["claims"])
            if valid
            else REFUSAL
        )
        return {"evidence_valid": valid, "answer": answer, "trace": state["trace"] + ["verify"]}

    def refuse(state: State):
        return {
            "claims": [],
            "evidence_valid": False,
            "answer": REFUSAL,
            "trace": state["trace"] + ["refuse"],
        }

    graph = StateGraph(State)
    for name, node in [
        ("plan", plan),
        ("retrieve", retrieve),
        ("quote", quote),
        ("verify", verify),
        ("refuse", refuse),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda s: "quote" if s["sources"] else "refuse",
        {"quote": "quote", "refuse": "refuse"},
    )
    graph.add_edge("quote", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What supply chain risks are disclosed?")
    parser.add_argument("--ticker", default="DEMO", choices=["DEMO", "OTHER"])
    args = parser.parse_args()
    result = build_graph().invoke({"question": args.question, "ticker": args.ticker})
    print("TRAINING DEMO: fictional documents, keyword search, no LLM\n")
    print(result["answer"])
    print("\nTrace:", " -> ".join(result["trace"]))
    print("\nSources:", json.dumps([asdict(c) for c in result["sources"]], indent=2))


if __name__ == "__main__":
    main()
