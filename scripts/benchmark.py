"""Compare rankers on passage-text labels for the locally indexed corpus."""

import argparse
import json
from pathlib import Path

from policypilot.config import Settings
from policypilot.embeddings import Embedder
from policypilot.store import LocalStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true")
    args = p.parse_args()
    s = Settings.load()
    db = LocalStore(s.data_dir / "index.sqlite", Embedder(s.embedding_model))
    cases = [json.loads(line) for line in Path("eval/pilot.jsonl").read_text().splitlines()]
    for mode in ["dense", "hybrid"] + (["reranked"] if args.rerank else []):
        db.hybrid = mode != "dense"
        if mode == "reranked":
            from sentence_transformers import CrossEncoder

            db.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        recalls = []
        rr = []
        for case in cases:
            with db.connect() as conn:
                gold = {
                    id
                    for id, payload in conn.execute(
                        "SELECT id,payload FROM chunks WHERE ticker=?", (case["ticker"],)
                    )
                    if case["gold_text"] in " ".join(json.loads(payload)["text"].split())
                }
            if not gold:
                raise ValueError("Gold evidence not found: " + case["ticker"])
            found = [
                i
                for i, h in enumerate(db.search(case["question"], case["ticker"]), 1)
                if h.passage.chunk_id in gold
            ]
            recalls.append(len(found) / len(gold))
            rr.append(1 / found[0] if found else 0)
        print(
            mode,
            "n=",
            len(cases),
            "recall@5=",
            sum(recalls) / len(cases),
            "MRR@5=",
            sum(rr) / len(cases),
            flush=True,
        )


if __name__ == "__main__":
    main()
