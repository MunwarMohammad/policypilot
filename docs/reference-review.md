# Review of the supplied project

Reviewed the supplied ZIP snapshot (archive comment identifies commit
`4ca031c0fe7f6031a6c1c88aea53f4bfe1c6f205`) and all six pages of the project overview.
This is a review of supplied files, not a claim that the live repository is unchanged.
Upstream: https://github.com/yashprogrammer/azure-databricks-agentic-qa

The structure is useful: ingestion, interchangeable retrieval backends, a LangGraph
workflow, Streamlit, evaluation, and deployment configuration. Several implementation
details should be improved before using it as an enterprise project:

1. `agent/graph.py` considers an answer grounded when retrieved results exist and any
   numeric bracket citation appears. It does not verify that the number identifies a
   retrieved source, that every claim is cited, or that the evidence supports the claim.
2. `eval/agent_eval.py` wraps search with a method that does not accept the `ticker`
   argument. The main retrieval tool passes that filter; the wrapper must preserve it.
3. The generic ingestion pipeline calls `store.upsert()`, while the Databricks Delta Sync
   backend intentionally raises `NotImplementedError` for upsert. Cloud ingestion uses
   the notebook instead; these are separate paths that need clear orchestration.
4. The overview's 1,200-word chunks exceed the embedding model's default token window.
   Chunk size must be aligned with the actual tokenizer to avoid silent truncation.
5. A fixed plan → retrieve → answer → verify graph is a controlled workflow. Add bounded
   decisions, tool selection or retrieval retries if claiming more autonomous behavior.
6. Credential-free deployment means no stored long-lived deployment credential; the
   external provider may still require an API key. Cloud authentication, permissions,
   evaluation and deployment need real verification before being presented as complete.

The documents' build instructions are reference material. No cloud deployment or
external model call has been executed as a result of reading them.
