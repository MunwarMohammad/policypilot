"""Build source notes from the same ordered hits used to number the answer."""


def citation_notes(state):
    for number, hit in enumerate(state.get("cited_hits", []), 1):
        p = hit.passage
        quotes = dict.fromkeys(c.quote for c in state["draft"].claims if c.source_id == p.chunk_id)
        yield (
            f"[{number}] {p.company} · Form 10-K · filed {p.filing_date} · {p.accession}",
            p.source_url,
            list(quotes),
        )
