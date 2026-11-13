"""Run with: uv run streamlit run app.py"""

import streamlit as st
from groq import BadRequestError

from policypilot.config import TICKERS, Settings
from policypilot.llm import provider_code

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
                    st.markdown(state["answer"].replace("$", r"\$"))
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

                    history = st.session_state.setdefault("filing_threads", {}).setdefault(
                        ticker, []
                    )
                    state = build_agent(
                        JSONModel(settings.backend), get_store(settings), settings.min_score
                    ).invoke({"question": question, "ticker": ticker, "history": history[-3:]})
                    if state["reason"] == "checks_passed":
                        history.append({"question": question, "answer": state["answer"]})
                    elif settings.backend == "local":
                        from policypilot.review import enqueue

                        ticket = enqueue(settings.data_dir, ticker, question, state)
                        st.caption(f"Queued for local human review (#{ticket}).")
                    from policypilot.presentation import citation_notes

                    st.markdown("**Answer**")
                    st.markdown(state["answer"].replace("$", r"\$"))
                    notes = list(citation_notes(state))
                    if notes:
                        st.markdown("**Citations**")
                    for label, url, quotes in notes:
                        st.write(label)
                        st.markdown(f"[Open SEC filing]({url})")
                        for quote in quotes:
                            st.text("“" + " ".join(quote.split()) + "”")
                with st.expander("Execution trace"):
                    st.write(" → ".join(state["trace"]))
        except BadRequestError as exc:
            known = {
                "context_length_exceeded": "The question and evidence exceeded the model context limit.",
                "model_decommissioned": "The configured Groq model is no longer available.",
                "json_validate_failed": "The model could not produce a valid structured answer.",
            }
            st.error(
                known.get(provider_code(exc), "Groq rejected the model request (HTTP 400).")
                + " No verified answer was produced."
            )
        except Exception as exc:
            # Avoid exposing credentials, provider response bodies or SDK request headers.
            st.error(
                f"Request failed ({type(exc).__name__}). Check .env, ingestion status, "
                "provider access, and the setup guide. No answer was generated."
            )
