"""Live evaluation separates mechanical checks from a fallible model support check."""

import argparse
import json
from pathlib import Path

from policypilot.agent import build_agent
from policypilot.config import Settings
from policypilot.llm import JSONModel
from policypilot.runtime import retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation.json"))
    args = parser.parse_args()
    settings = Settings.load()
    graph = build_agent(JSONModel(settings.backend), retrieval(settings), settings.min_score)
    results = []
    for line in args.dataset.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        state = graph.invoke({"question": item["question"], "ticker": item["ticker"]})
        answered = state["reason"] == "checks_passed"
        expected = item.get("expected_refusal")
        results.append(
            {
                **item,
                "answer": state["answer"],
                "answered": answered,
                "citation_valid": state.get("citation_valid", False),
                "model_support_check": state.get("support_check_passed", False),
                "refusal_correct": None if expected is None else (not answered) == expected,
                "retrieved_ids": [h.passage.chunk_id for h in state["hits"]],
                "trace": state["trace"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "results": results,
                "note": "Model support checks are not human-labeled accuracy. Review answers manually.",
            },
            indent=2,
        )
    )
    print(f"Evaluated {len(results)} questions. Report: {args.output}")


if __name__ == "__main__":
    main()
