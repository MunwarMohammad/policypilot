# From annual report to answer

The input to this application is a question, not a ticker-specific prompt template. The planning node resolves a company when possible and suggests a search phrase. The retriever uses the same interface whether its index is on the laptop or in Databricks.

## What persists

- Original plain-text filings and a manifest identify the source documents.
- Chroma stores local passage embeddings. Cloud search uses an index synchronized from a Delta table.
- Chat turns live in the browser session. They are displayed as history; each new question is planned independently.

## What the reader sees

The answer is followed by numbered sources with company, filing date, accession and an excerpt. No retrieved passage means no substantive answer. If a provider rejects a request for size, the workflow drops the lowest-ranked passage and retries. This preserves numbering for the remaining evidence.

## What requires a workspace

Unity Catalog permissions, the Delta Sync index, model gateway access, hosted evaluation and bundle deployment must be exercised in the target workspace. Example configuration does not establish that any of these resources exist.
