# Azure Databricks deployment

This is an operator-run setup guide. Nothing in this repository automatically creates an
Azure subscription, workspace, catalog, external model provider or access grants. Cloud
execution is unverified until the following steps pass in your workspace.

## 1. Establish workspace prerequisites

Use an Azure subscription and a workspace with Unity Catalog, serverless availability,
Databricks Apps, and the required search/gateway services in its region. Confirm costs
before provisioning. Create a dev catalog and schema and a UC Volume for the export.
Configure local SDK authentication with the Databricks CLI's OAuth login/profile flow.
Do not commit a token or service-principal secret.

Example resource names used below are defaults you can change, not existing resources:

```sql
CREATE CATALOG IF NOT EXISTS policypilot_dev;
CREATE SCHEMA IF NOT EXISTS policypilot_dev.filings;
CREATE VOLUME IF NOT EXISTS policypilot_dev.filings.raw;
```

An administrator may need to create the catalog or configure its managed storage.

## 2. Populate Delta

```bash
uv sync --frozen --extra semantic --extra databricks --group dev
uv run policypilot ingest --tickers AAPL
uv run policypilot export --output data/chunks.jsonl
```

Upload `data/chunks.jsonl` to `/Volumes/policypilot_dev/filings/raw/chunks.jsonl`.
Import and run `notebooks/load_delta.py` on Unity Catalog-enabled compute. Set its widgets
for your table and input path. It creates a CDF-enabled Delta table and atomically replaces
rows for each company included in the export. Export complete company snapshots: partial
company exports would remove omitted rows. Do not mix embedding models in one table/index.
This notebook expects MiniLM's 384-dimensional embeddings. Change model, notebook and
index together when experimenting with a different model.

## 3. Create the search index

The current Databricks documentation calls this service AI Search; this repository uses
the compatible Vector Search SDK interface. The adapter targets STANDARD endpoints and
self-managed embeddings, not storage-optimized SQL-filter syntax.

```bash
uv run --extra databricks python scripts/create_vector_index.py \
  --endpoint policypilot-search \
  --table policypilot_dev.filings.chunks \
  --index policypilot_dev.filings.chunks_index \
  --create-endpoint
```

Run creation only once. On later data updates obtain the index with `get_index`, call
`sync()`, and wait for readiness using `wait_until_ready(wait_for_updates=True)` before
querying. A triggered index is not automatically fresh immediately after table writes.
See [index setup](https://learn.microsoft.com/en-us/azure/databricks/vector-search/create-vector-search).

## 4. Configure the model service

Create a Unity Gateway model service in your region using a supported Databricks model
or an external provider. Configure external provider credentials through workspace secret
management; do not put them into bundle variables or Git. Record its three-part service
name. Grant the app identity `USE CATALOG`, `USE SCHEMA` and `EXECUTE` as appropriate.
The adapter calls `/ai-gateway/mlflow/v1/chat/completions` and validates returned JSON
locally; verify that your chosen service supports this API and the requested model output.
See [Unity model services](https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/model-services).

## 5. Deploy the app

Set `DATABRICKS_HOST` and these bundle variables in your shell:

```bash
export BUNDLE_VAR_vector_endpoint=policypilot-search
export BUNDLE_VAR_vector_index=policypilot_dev.filings.chunks_index
export BUNDLE_VAR_model_service=policypilot_dev.filings.research_model

databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run policypilot_app -t dev
```

Use a current CLI with `apps.config` support. Bundle config supplies the app command and
environment; a separate app.yaml is not needed. `requirements.txt` installs the package
with semantic and Databricks dependencies. Model downloads require allowed outbound
access and enough app memory; pre-cache the model if network rules require it.

After app creation, find its assigned service principal. Grant only the data/index and
model-service privileges it needs, including catalog/schema use and index SELECT.
App service identity access is separate from human workspace access. Configure who can
open the app. Confirm both successful requests and rejected unauthorized identities.
Test citation links, no-evidence refusal and company isolation from the deployed app.
Do not claim a successful deployment from bundle creation alone.

## 6. GitHub OIDC deployment

Create a GitHub environment named `dev`. Configure a Databricks service-principal
federation policy restricted to this repository/environment's exact OIDC subject. Subject
formats can vary; use the actual repository claim rather than copying an example.
Set these environment variables in GitHub:

- `DATABRICKS_HOST`
- `DATABRICKS_CLIENT_ID`
- `PP_VECTOR_ENDPOINT`
- `PP_VECTOR_INDEX`
- `PP_MODEL_SERVICE`

Run the manual deployment workflow. It uses `github-oidc` authentication and does not need
stored Databricks client secrets or a PAT. Pin the setup actions to reviewed commit SHAs
before production adoption. See [federation setup](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/provider-github).

The deployment workflow deploys an app against preconfigured infrastructure. It does not
run ingestion, create the gateway, grant permissions, or execute live evaluation. Those
steps remain explicit so a GitHub push cannot silently provision billable infrastructure.
