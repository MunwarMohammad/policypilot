# Implementation and validation status

The source project now implements the planned local pipeline and cloud adapters. The
original learning demo remains available as `uv run policypilot demo`.

| Area | Implementation | Remaining validation/configuration |
|---|---|---|
| SEC ingestion | Client, cache, retries, metadata, report | Real identifying user-agent and live download |
| Chunking and retrieval | Model token windows, MiniLM, SQLite cosine ranking | Corpus-specific retrieval quality calibration |
| Agent | Planner, bounded retry, structured claims, evidence gate | Real provider key and live behavior evaluation |
| Interface | Streamlit demo and SEC research | User testing and cloud access setup |
| Evaluation | Offline tests and live question runner | Human-labeled ground truth and measured results |
| Delta and Vector Search | Notebook, provisioning script, query adapter | Workspace, grants, actual table/index execution |
| Unity Gateway | SDK authenticated model-service client | Available service and model privileges |
| GitHub | CI, manual OIDC deployment, publishing guide | User repository and federation policy |

Start with one real filing, inspect its retrieved passages, and add labeled questions
before scaling to the configured 20 companies. Production work includes observability,
independent support judging, prompt-injection evaluation, historical version selection,
load testing, disaster recovery, and automated ingestion scheduling.
