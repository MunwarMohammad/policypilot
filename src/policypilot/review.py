"""Local operator review; never publishes unverified claims to the chat."""

import argparse
import json
import sqlite3


def connection(root):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "review.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS review(id INTEGER PRIMARY KEY, ticker TEXT, question TEXT, evidence TEXT, status TEXT DEFAULT 'pending', note TEXT)"
    )
    return db


def enqueue(root, ticker, question, state):
    with connection(root) as db:
        return db.execute(
            "INSERT INTO review(ticker,question,evidence) VALUES(?,?,?)",
            (ticker, question, json.dumps([h.passage.model_dump() for h in state["hits"]])),
        ).lastrowid


def main():
    from policypilot.config import Settings

    p = argparse.ArgumentParser()
    p.add_argument("--resolve", type=int)
    p.add_argument("--note")
    args = p.parse_args()
    with connection(Settings.load().data_dir) as db:
        if args.resolve:
            if not args.note:
                p.error("Resolution note required")
            if not db.execute(
                "UPDATE review SET status='resolved',note=? WHERE id=? AND status='pending'",
                (args.note, args.resolve),
            ).rowcount:
                p.error("Pending record not found")
        else:
            for row in db.execute(
                "SELECT id,ticker,question,evidence FROM review WHERE status='pending'"
            ):
                print(*row, sep="\n")


if __name__ == "__main__":
    main()
