import argparse
import json
import os
import sys
from pathlib import Path

from policypilot.config import TICKERS, Settings


def main():
    parser = argparse.ArgumentParser(description="PolicyPilot SEC filing research")
    commands = parser.add_subparsers(dest="command")
    demo = commands.add_parser("demo", help="Offline fictional-data lesson")
    demo.add_argument("question", nargs="?", default="supply chain")
    demo.add_argument("--ticker", default="DEMO", choices=["DEMO", "OTHER"])
    ingest = commands.add_parser("ingest", help="Download, chunk and index latest 10-Ks")
    ingest.add_argument("--tickers", nargs="+", default=["AAPL"], choices=TICKERS)
    ingest.add_argument("--all", action="store_true")
    ask = commands.add_parser("ask", help="Run live RAG with configured model")
    ask.add_argument("question")
    ask.add_argument("--ticker", required=True, choices=TICKERS)
    export = commands.add_parser("export", help="Export indexed rows for the Delta notebook")
    export.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "demo":
        from policypilot.demo import build_graph

        state = build_graph().invoke({"question": args.question, "ticker": args.ticker})
        print("FICTIONAL DATA / KEYWORD RETRIEVAL / NO LLM\n")
        print(state["answer"])
        print("Trace:", " -> ".join(state["trace"]))
        return
    settings = Settings.load()
    try:
        from policypilot.runtime import retrieval

        store = retrieval(settings)
        if args.command in {"ingest", "export"} and settings.backend != "local":
            raise ValueError("Ingest/export locally, then load the export with the Delta notebook")
        if args.command == "ingest":
            from policypilot.ingestion import ingest
            from policypilot.sec import SECClient

            client = SECClient(os.getenv("SEC_USER_AGENT", ""), settings.data_dir / "cache")
            report = ingest(
                TICKERS if args.all else args.tickers,
                client,
                store.embedder,
                store,
                settings.data_dir,
            )
            print(json.dumps(report, indent=2))
            if any(r["status"] != "ok" for r in report):
                sys.exit(1)
        elif args.command == "export":
            rows = store.export()
            if not rows:
                raise ValueError("No indexed documents to export")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            print(f"Exported {len(rows)} chunks to {args.output}")
        elif args.command == "ask":
            from policypilot.agent import build_agent
            from policypilot.llm import JSONModel

            state = build_agent(JSONModel(settings.backend), store, settings.min_score).invoke(
                {"question": args.question, "ticker": args.ticker}
            )
            print(state["answer"])
            print("\nTrace:", " -> ".join(state["trace"]))
            for i, hit in enumerate(state["hits"], 1):
                p = hit.passage
                print(f"[{i}] {p.ticker} | report {p.report_date} | filed {p.filing_date}")
                print(p.source_url)
    except (ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Configuration/data error: {exc}\n")
