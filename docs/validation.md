# Validation performed

Verified locally on October 24, 2026:

- 27 offline tests passed, including Streamlit AppTest and a mocked Databricks REST response.
- Ruff lint and formatting checks passed.
- Python wheel and source distribution built successfully.
- The actual all-MiniLM-L6-v2 model produced eight token-bounded chunks from synthetic
  test text; every chunk fit the model limit. A paraphrased security query retrieved the
  cybersecurity passage (cosine score approximately 0.339). This is a smoke test, not an
  accuracy benchmark.
- The offline CLI returned a cited training quote and refused an unrelated question.

Not performed: live SEC download with a user-provided identifying contact, Groq inference,
full 20-company ingestion, human-labeled evaluation, Azure provisioning, Delta notebook
execution, cloud authentication, deployed app smoke test, or a deployed application smoke test.

The implementation includes these paths/configurations, but local tests do not establish
that the external services or permissions are configured correctly.

Update on November 2, 2026: GitHub reports the CI workflow completed successfully.
The cloud deployment workflow has not been verified.

### Local integration run — November 12, 2026

All 27 tests passed; the lockfile, lint, formatting, wheel and source build passed.
The offline CLI returned both a cited answer and an unsupported-question refusal.

Using the [Microsoft 2025 Annual Report](https://www.microsoft.com/investor/reports/ar25/index.html)
from the issuer, the production HTML cleaner, tokenizer, MiniLM embedder and SQLite store
produced 207 passages. After reopening the index, a cloud-revenue question retrieved
relevant Microsoft passages; selecting an unindexed company returned none. All 207
exported JSONL rows had distinct IDs, the expected source URL and 384 embedding dimensions.
The LangGraph answer path and Streamlit research interface displayed the cited result
using deterministic model responses. The evaluation CLI wrote all five results using
the real Groq SDK with simulated HTTP responses. These are integration checks, not
measurements of LLM accuracy. Test inputs, index and logs are isolated in tmp/local-validation.

Live EDGAR discovery/download and Groq inference were not tested: SEC_USER_AGENT and
GROQ_API_KEY were unset. The issuer report used fixture filing identifiers, not SEC-verified
metadata. No Azure services were invoked.
