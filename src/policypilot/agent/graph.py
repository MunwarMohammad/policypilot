"""LangGraph agent: plan -> retrieve -> answer -> verify.

Genuinely agentic rather than a bare RAG chain: the plan step decides what to search for
and whether a structured metadata lookup applies, and the verify step is a hard gate —
an answer without a citation back to a retrieved chunk is replaced with a refusal instead
of being returned to the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from policypilot.agent.llm import LLMClient
from policypilot.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CITATION_RE,
    PLAN_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
)
from policypilot.agent.tools import filing_metadata, format_context, retrieve
from policypilot.config import DEFAULT_TICKERS
from policypilot.retrieval.base import SearchResult, VectorStore


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    ticker_hint: str | None
    results: list[SearchResult]
    metadata: dict | None
    draft_answer: str
    final_answer: str
    grounded: bool


@dataclass
class FilingWorkflow:
    llm: LLMClient
    store: VectorStore

    def plan_node(self, state: AgentState) -> AgentState:
        reply = self.llm.complete(
            system=PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["question"]}],
        ).strip()
        try:
            parsed = json.loads(reply)
            query = str(parsed.get("search_query") or "").strip()
            ticker = parsed.get("ticker")
            ticker = str(ticker).strip().upper() if ticker else None
        except (json.JSONDecodeError, AttributeError):
            query, ticker = reply, None
        ticker_hint = ticker if ticker in DEFAULT_TICKERS else None
        return {"search_query": query or state["question"], "ticker_hint": ticker_hint}

    def retrieve_node(self, state: AgentState) -> AgentState:
        ticker_hint = state.get("ticker_hint")
        results = retrieve(self.store, state["search_query"], k=5, ticker=ticker_hint)
        metadata = filing_metadata(ticker_hint) if ticker_hint else None
        return {"results": results, "metadata": metadata}

    def answer_node(self, state: AgentState) -> AgentState:
        results = state["results"]
        while results:
            system = ANSWER_SYSTEM_PROMPT.format(context=format_context(results))
            try:
                draft = self.llm.complete(
                    system=system, messages=[{"role": "user", "content": state["question"]}]
                )
            except Exception as error:
                # Drop only the lowest-ranked passage; citation numbers remain stable.
                if getattr(error, "status_code", None) != 413 or len(results) == 1:
                    raise
                results = results[:-1]
                continue
            return {"draft_answer": draft, "results": results}
        return {"draft_answer": REFUSAL_MESSAGE}

    def verify_node(self, state: AgentState) -> AgentState:
        draft = state["draft_answer"]
        grounded = bool(state.get("results")) and bool(CITATION_RE.search(draft))
        final = draft if grounded else REFUSAL_MESSAGE
        return {"final_answer": final, "grounded": grounded}


def build_graph(llm: LLMClient, store: VectorStore):
    steps = FilingWorkflow(llm, store)
    graph = StateGraph(AgentState)
    nodes = [
        ("plan", steps.plan_node),
        ("retrieve", steps.retrieve_node),
        ("answer", steps.answer_node),
        ("verify", steps.verify_node),
    ]
    for name, operation in nodes:
        graph.add_node(name, operation)
    graph.set_entry_point(nodes[0][0])
    names = [name for name, _ in nodes]
    for current, following in zip(names, [*names[1:], END], strict=True):
        graph.add_edge(current, following)
    return graph.compile()


def ask(llm: LLMClient, store: VectorStore, question: str) -> AgentState:
    app = build_graph(llm, store)
    return app.invoke({"question": question})
