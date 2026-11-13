# SEC Annual Filing Research with LangGraph

[![CI](https://github.com/MunwarMohammad/policypilot/actions/workflows/ci.yml/badge.svg)](https://github.com/MunwarMohammad/policypilot/actions/workflows/ci.yml)

Ask a question about a company’s latest indexed 10-K. The app answers first, then links each numbered citation to a supporting quote and its SEC filing.

![Local Apple filing answer with numbered SEC citations](docs/evidence/filing-answer.png)

| Local run | Result |
|---|---:|
| Companies indexed | 20 |
| Annual filings | 20 |
| Searchable passages | 8,588 |

[Per-company ingestion evidence](docs/evidence/ingestion.csv). These are corpus counts, not accuracy metrics. Azure adapters are implemented but have not been deployed or validated.

## Quick start

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --group dev
uv run policypilot demo
uv run streamlit run app.py
```

Choose **Offline demo** in the UI for fictional documents and keyword retrieval without keys.
This is distinct from the real semantic RAG path below.

## Real filings and semantic RAG

```bash
uv sync --frozen --extra semantic --group dev
cp .env.example .env
# Edit .env: set a real SEC_USER_AGENT contact and your GROQ_API_KEY.
uv run policypilot ingest --tickers AAPL
uv run policypilot ask "What supply chain risks are disclosed?" --ticker AAPL
uv run streamlit run app.py
```

Choose **SEC filing research** in the UI. The embedding model downloads on first use.
Groq receives the question and retrieved public filing passages. The selected model must
be available to your account. Data stays under the ignored `data/` directory locally.

```bash
uv run policypilot ingest --all
uv run python -m policypilot.evaluation
```

Ingestion reports each company independently and exits unsuccessfully if any company
fails. See `data/ingestion-report.json`. A failed refresh preserves previously indexed
passages. One latest successfully ingested 10-K is retained per company. The selected
company is an explicit UI/CLI filter, not an LLM guess. Multi-company comparisons and
historical year selection are not implemented.

## Architecture

```mermaid
flowchart TD
    SEC[SEC EDGAR] --> Raw[Cached HTML and filing metadata]
    Raw --> Clean[Clean and tokenize]
    Clean --> Chunk[Overlapping token windows]
    Chunk --> Embed[MiniLM embeddings]
    Embed --> Local[SQLite local vector store]
    Local --> Export[JSONL export]
    Export --> Delta[Unity Catalog Delta table]
    Delta --> Index[Delta Sync Vector Search index]
    Q[Question + selected company] --> Plan[LLM search planner]
    Plan --> Retrieve[Company-filtered retrieval]
    Local --> Retrieve
    Index --> Retrieve
    Retrieve -->|no evidence; retry once| Retry[Original question]
    Retry --> Retrieve
    Retrieve -->|evidence| Draft[Structured answer claims]
    Retrieve -->|retry exhausted| Refuse[Refusal]
    Draft --> Verify[Source IDs + exact quotes + model support check]
    Verify -->|pass| Answer[Cited answer and source links]
    Verify -->|fail| Refuse
```

## What's implemented

- SEC ticker resolution, older-submission fallback, request throttling, bounded retries,
  immutable document caching, provenance and per-company ingestion reports.
- Token-aware chunking sized to the embedding model; stable content-derived chunk IDs.
- Atomic company refreshes and cosine search using SQLite/NumPy locally. This exhaustive
  search is suitable for a portfolio corpus, not a distributed enterprise search engine.
- LangGraph planning, bounded retry, structured generation, citation validation, a separate
  LLM support check, refusals and visible execution traces.
- Streamlit question interface with document text, dates, accessions and SEC links.
- Evaluation runner, offline tests, dependency lockfile, CI, manual OIDC deployment workflow.
- Databricks retrieval/gateway adapters, Delta loading notebook, index provisioning script,
  and app deployment bundle. Setup is described in [deployment](docs/deployment.md).

## Validation and limitations

```bash
uv run pytest -q
uv run ruff check src tests app.py notebooks scripts
uv run ruff format --check src tests app.py notebooks scripts
uv build
```

Tests use deterministic doubles for external services. They verify graph routing, refusal,
company isolation, citation rejection, SEC fallback/retries, and atomic storage behavior.
Live evaluation results are written only when you run the evaluation command. Starter
questions are not a human-labeled accuracy benchmark. Add source-specific expected
passages and human-reviewed answers before reporting retrieval recall or answer accuracy.

A valid quote does not prove that a claim is true. The support-check model can also be
wrong, particularly when judging its own generation. Inspect sources. Retrieved content
is treated as untrusted input, but prompt injection is not completely solved. The app is
for filing research, not investment predictions. HTML table extraction is basic and may
lose layout; use structured XBRL for precise financial computations in a future extension.

No app-level user authorization is implemented. Databricks production access must be
configured through app access, service-principal permissions, catalog/index grants and
model-service permissions. The company selector is a retrieval filter, not an ACL.

## Project map

```text
app.py                        Streamlit interface
src/policypilot/
  sec.py                      EDGAR acquisition and HTML cleaning
  embeddings.py               Token-aware chunks and embeddings
  store.py                    Atomic local vector store
  ingestion.py                Per-company ingestion orchestration
  agent.py                    LangGraph workflow and evidence checks
  llm.py                      Groq / Unity Gateway model calls
  databricks_backend.py       Managed vector search adapter
  cli.py                      Demo, ingestion, question and export commands
  evaluation.py               Live evaluation report
  demo.py                     Offline learning example
notebooks/load_delta.py      Export-to-Delta ingestion
scripts/create_vector_index.py
.github/workflows/           Offline CI and manual cloud deployment
```

See [design decisions](docs/design-decisions.md) and [Azure deployment instructions](docs/deployment.md).

## Acknowledgements

The architecture was informed by [azure-databricks-agentic-qa](https://github.com/yashprogrammer/azure-databricks-agentic-qa). This implementation uses token-aware chunking, a SQLite vector store, and claim-level quote and support checks.

## Local research features

Set `PP_HYBRID=1` in .env for BM25 + dense cosine retrieval combined by reciprocal rank fusion. Add `PP_RERANK=1` to rescore up to 20 candidates per query with a cross-encoder. Restart Streamlit after changing retrieval options. These switches affect local search; the existing cosine floor still applies after ranking.

The model selects a scoped filing search or multi-query overview through the chat tool API. Streamlit passes up to three previous successful turns for the same ticker to resolve follow-ups. Browser-session context is not durable user memory. Local UI refusals enter a SQLite review queue. Inspect with `uv run python -m policypilot.review`; resolve with `--resolve ID --note "review findings"`. Resolution records an operator decision; it does not auto-publish an answer.

Reproduce the two-question passage-label smoke benchmark with `uv run --extra semantic python scripts/benchmark.py --rerank`. Dense recall@5 / MRR@5: 0.50 / 0.50; hybrid: 1.00 / 0.625; hybrid + reranking: 1.00 / 0.625. This tiny development set does not demonstrate a general quality gain.
