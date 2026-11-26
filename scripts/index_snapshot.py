"""Populate local Chroma from the filings identified by the bundled manifest."""
from policypilot.config import RAW_DIR
from policypilot.ingestion.manifest import load_manifest
from policypilot.ingestion.pipeline import index_text
from policypilot.retrieval.local_chroma import LocalChromaVectorStore


def main():
    store = LocalChromaVectorStore()
    for ticker, filing in load_manifest().items():
        path = RAW_DIR / f"{ticker}_{filing['accession_number']}.txt"
        count = index_text(ticker, filing, path.read_text(), store)
        print(f"{ticker}: {count} passages", flush=True)


if __name__ == "__main__":
    main()
