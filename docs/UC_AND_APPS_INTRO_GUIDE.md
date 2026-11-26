# Cloud components and permissions

| Component | Role | First check |
|---|---|---|
| Unity Catalog | Names and controls access to tables, indexes and model services | Can the app principal use the catalog and schema? |
| Delta table | Holds filing passages, metadata and vectors | Are the expected tickers and embedding dimensions present? |
| Vector Search | Retrieves passages from a synchronized index | Is the index ready after loading or refreshing the table? |
| AI Gateway | Routes model requests through a governed service | Can the app identity invoke the service without accessing the provider key? |
| Databricks App | Runs the Streamlit entry point | Does a user question produce both an answer and numbered sources? |

Use separate identities for provisioning and runtime. Validate the table, index, model and app in that order: a working Streamlit page alone verifies none of the upstream permissions. See the [deployment runbook](DEPLOYMENT_GUIDE.md).
