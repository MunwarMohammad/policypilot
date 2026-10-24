import json
from datetime import UTC, datetime

from policypilot.sec import clean_html


def ingest(tickers, client, embedder, store, data_dir):
    """One active latest annual filing per company; partial failures are reported."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    report = []
    for ticker in tickers:
        try:
            filing = client.latest(ticker)
            html = client.download(filing)
            (raw / f"{ticker}-{filing.accession}.html").write_text(html)
            text = clean_html(html)
            if not text.strip():
                raise ValueError("Filing contains no extractable text")
            passages = embedder.chunks(text, filing)
            store.replace_company(ticker, passages)
            report.append(
                {
                    "ticker": ticker,
                    "status": "ok",
                    "chunks": len(passages),
                    "filing": filing.model_dump(),
                }
            )
        except Exception as exc:
            report.append({"ticker": ticker, "status": "failed", "error": str(exc)})
    manifest = {"created_at": datetime.now(UTC).isoformat(), "results": report}
    (data_dir / "ingestion-report.json").write_text(json.dumps(manifest, indent=2))
    return report
