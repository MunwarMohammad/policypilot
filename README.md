# SEC Filing Research

Ask a question about an annual report, read the answer, and open **Citations** to inspect the numbered passages. PolicyPilot runs the same four-step LangGraph workflow against local or Databricks services.

## Run locally

Use Python 3.11+ and uv. Start with the example environment file only if you do not already have a private configuration.

```sh
uv sync --frozen --extra dev
test -f .env || cp .env.example .env
# Set PP_ENV=local, GROQ_API_KEY and SEC_EDGAR_USER_AGENT in .env.
# Include a real contact email in your SEC user agent.
uv run python scripts/index_snapshot.py  # use bundled filings
# To refresh from EDGAR: uv run python -m policypilot.ingestion.pipeline
uv run streamlit run src/policypilot/app/streamlit_app.py
```

The bundled filing text is not a searchable index by itself. Local ingestion writes filing metadata and populates Chroma using MiniLM embeddings. The SQLite index from the earlier implementation is not a substitute for Chroma.

## Follow a question through the code

| Step | Behavior |
|---|---|
| Plan | The model rewrites the question and resolves a known ticker if possible. |
| Retrieve | Search up to five passages, applying the ticker filter when present. |
| Answer | Give the model numbered passages and ask for a citation on each factual claim. |
| Verify | Return a refusal when retrieval yielded nothing or the answer lacks a numeric citation marker. |

The final check verifies citation **presence**. It cannot establish that every statement is true or that every number refers to a valid passage. The source panel is there for inspection.

The graph and its prompts live under `src/policypilot/agent/`. Ingestion and vector-store implementations have separate packages; chat history belongs to the Streamlit session. For cloud setup, read the [deployment guide](docs/DEPLOYMENT_GUIDE.md) and adjust workspace-specific examples before running a bundle. This checkout has not been deployed to Azure.

## Verification and provenance

Run `uv run pytest -q` for the behavior suite. Retrieval tests use actual local embeddings and temporary Chroma collections; they do not need a Groq key. Model downloads require connectivity on the first run.

Adapted from [azure-databricks-agentic-qa](https://github.com/yashprogrammer/azure-databricks-agentic-qa/tree/471c4230bc0f3fbe70bb6c4fd72c098bd32778c6). Filing text, evaluation inputs, model prompts and external integration contracts retain their upstream roles.
