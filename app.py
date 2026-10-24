"""Run with: uv run streamlit run app.py"""

import streamlit as st

from policypilot.config import TICKERS, Settings

st.set_page_config(page_title="PolicyPilot", page_icon="📑", layout="wide")
st.title("PolicyPilot")
st.caption("Research SEC annual filings with inspectable evidence")
mode = st.sidebar.radio("Mode", ["Offline demo", "SEC filing research"])

if mode == "Offline demo":
    st.info("Fictional training documents • keyword search • no LLM or cloud account required")
    ticker = st.sidebar.selectbox("Company", ["DEMO", "OTHER"])
else:
    ticker = st.sidebar.selectbox("Company", TICKERS)
    st.sidebar.caption(
        "Ingest this company before asking questions. Each company uses its latest "
        "successfully ingested 10-K, which may not be today's latest filing."
    )
    st.caption("Research support only. Inspect sources; model validation can be wrong.")
question = st.chat_input("Ask about supply chain, cybersecurity, or another disclosed risk")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding and checking evidence…"):
                if mode == "Offline demo":
                    from policypilot.demo import build_graph

                    state = build_graph().invoke({"question": question, "ticker": ticker})
                    st.write(state["answer"])
                    for source in state["sources"]:
                        with st.expander(source.id + " — " + source.source):
                            st.write(source.text)
                else:
                    from policypilot.agent import build_agent
                    from policypilot.llm import JSONModel
                    from policypilot.runtime import retrieval

                    settings = Settings.load()

                    # Cache only public retrieval resources, never user questions or answers.
                    @st.cache_resource
                    def get_store(settings):
                        return retrieval(settings)

                    state = build_agent(
                        JSONModel(settings.backend), get_store(settings), settings.min_score
                    ).invoke({"question": question, "ticker": ticker})
                    st.write(state["answer"])
                    for i, hit in enumerate(state["hits"], 1):
                        p = hit.passage
                        with st.expander(f"[{i}] {p.company} — report date {p.report_date}"):
                            st.write(p.text)
                            st.caption(f"Filed {p.filing_date} · Accession {p.accession}")
                            st.link_button("Open SEC filing", p.source_url)
                with st.expander("Execution trace"):
                    st.write(" → ".join(state["trace"]))
        except Exception as exc:
            # Avoid exposing credentials, provider response bodies or SDK request headers.
            st.error(
                f"Request failed ({type(exc).__name__}). Check .env, ingestion status, "
                "provider access, and the setup guide. No answer was generated."
            )
