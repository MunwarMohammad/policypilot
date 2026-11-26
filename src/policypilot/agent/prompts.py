"""Stable model and response contracts shared by the graph and evaluations."""

import re

from policypilot.config import DEFAULT_TICKERS

CITATION_RE = re.compile(r"\[\d+\]")

PLAN_SYSTEM_PROMPT = (
    "You turn a user's question about SEC 10-K filings into a short search query "
    "for a semantic search index over filing text, and detect which company (if any) "
    "the question is about. Known companies (by ticker): " + ", ".join(DEFAULT_TICKERS) + ". "
    "The question may name a company by its ticker symbol OR by its common name (e.g. "
    "'Apple' means AAPL) — resolve either form to the ticker symbol.\n\n"
    "Reply with ONLY a JSON object of the form "
    '{"search_query": "<short search query text>", "ticker": "<TICKER or null>"}, '
    "nothing else."
)

ANSWER_SYSTEM_PROMPT = (
    "You are PolicyPilot, a compliance research assistant that answers questions about "
    "SEC 10-K filings using ONLY the numbered context provided below. This is a strict "
    "citation requirement: every sentence containing a factual claim MUST end with a "
    "bracketed citation number like [1] or [2] matching the context item it came from — "
    "no other citation style (no footnotes, no parentheses, no 'Source:' lines). "
    "Example: 'The filing discloses reliance on a limited number of suppliers [1].' "
    "If the context does not contain enough information to answer, say so plainly instead "
    "of guessing — never answer from general knowledge.\n\nContext:\n{context}"
)

REFUSAL_MESSAGE = (
    "I don't have enough grounded information in the ingested filings to answer that "
    "confidently. Try rephrasing, or ask about one of the ingested companies."
)
