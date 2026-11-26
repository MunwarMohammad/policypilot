# Deploy to a development workspace

1. Choose an Azure Databricks workspace with Unity Catalog, Apps, Vector Search and the required model service support. Set your CLI profile or DATABRICKS_HOST. Supply BUNDLE_VAR_admin_user for your own administrator. Review every catalog, endpoint and service name in the bundle, notebook and Python configuration; all must refer to your resources.
2. Prepare the catalog and schema. Review scripts/setup_unity_catalog.py before execution. Import notebooks/seed_chunks_table.py, set the sec_user_agent widget and check its destination table. Run the notebook on authorized compute; verify row counts, ticker coverage and embedding dimensions before syncing.
3. Create the Vector Search endpoint and Delta Sync index over the chunks table using the chunk ID as primary key and the existing embedding column. Wait for readiness and exercise a ticker-filtered query.
4. Configure the provider secret in Azure Key Vault. Inspect resources/ai_gateway.yml and configure the bundle's Groq key through the deployment environment. Grant only the required catalog, schema, index and model privileges to the app's identity.
5. Validate the target with `databricks bundle validate -t dev`, then deploy with `databricks bundle deploy -t dev`. Check the app startup command in app.yaml and start/deploy the app through the workspace. Staging and production placeholders must be replaced before using those targets.
6. Open the app as an intended user. Ask a filing question, compare the numbered source with the filing, and verify an unsupported question is refused. Record the actual run and permissions separately for each environment.

The manual GitHub workflow in .github/workflows/cd.yml expects a federated Azure identity, Key Vault access and workspace configuration. Review it before dispatch. Monitor endpoint and compute costs; tear down unused development resources.
