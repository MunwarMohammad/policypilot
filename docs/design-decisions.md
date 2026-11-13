# Why the filing pipeline works this way

Each passage retains the filing accession, period, company and URL. The tokenizer sets the window size so the embedding represents the whole passage. Overlap preserves context when a sentence crosses a boundary.

The selected ticker filters candidates before local cosine ranking. This keeps a relevant-sounding passage from a different company out of the answer. SQLite lets a company refresh replace its rows in one transaction.

Retrieval similarity measures proximity, not truth. Each generated claim therefore needs a quote from a retrieved passage and a separate support judgment. Claims that fail are withheld. When none survive, the agent can retry with a narrower quote, then refuse. A model judgment can still be wrong: the interface shows the evidence so readers can inspect it.

Local ingestion and QA are distinct from cloud deployment. The Delta notebook loads exported embeddings; it does not perform the original SEC download or embedding step.
