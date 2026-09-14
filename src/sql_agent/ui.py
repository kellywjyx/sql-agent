import os

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SQL Agent", page_icon="🗃️", layout="wide")
st.title("Read-only SQL Agent")
st.caption("SQLite · evaluated V5 default · optional V7 semantic grounding")
pipeline_profile = st.sidebar.selectbox("Pipeline", ["v5", "v7_grounded"])
schema_mode = st.sidebar.selectbox("Schema retrieval", ["auto", "full", "hybrid"])
model_profile = st.sidebar.selectbox("Local model", ["default", "qwen_v5", "arctic_sql", "xiyan_sql"])
generation_strategy = st.sidebar.selectbox("Generation", ["single", "adaptive_two"])
value_mode = st.sidebar.selectbox("Value context", ["probe", "profiled"])
question = st.text_input("Ask about the shop", "What is the total revenue from completed orders?")
if st.button("Run query", type="primary"):
    st.session_state.pop("result", None)
    with st.spinner("Generating and checking SQL…"):
        try:
            response = httpx.post(os.getenv("SQL_AGENT_API", "http://127.0.0.1:8102") + "/query",
                                  json={"question": question, "schema_mode": schema_mode,
                                        "model_profile": model_profile,
                                        "generation_strategy": generation_strategy,
                                        "value_mode": value_mode,
                                        "pipeline_profile": pipeline_profile}, timeout=600)
            response.raise_for_status()
            st.session_state.result = response.json()
        except httpx.HTTPError as exc:
            st.error(exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc))
if result := st.session_state.get("result"):
    if result.get("sql"):
        st.code(result["sql"], language="sql")
        st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), hide_index=True)
        st.caption(f"{result['row_count']} rows. Display truncated: {result.get('display_truncated', False)}")
    else:
        st.warning(result.get("error", "No valid query"))
    with st.expander("Execution and correction trace", expanded=True):
        st.json(result["attempts"])
    with st.expander("Candidate selection and semantic checks", expanded=True):
        st.json({"candidates": result.get("candidates"), "selection_reason": result.get("selection_reason"),
                 "semantic_risk": result.get("semantic_risk"), "needs_review": result.get("needs_review"),
                 "semantic_checks": result.get("semantic_checks"), "critic": result.get("critic"),
                 "candidate_result_group": result.get("candidate_result_group"),
                 "repair_events": result.get("repair_events")})
    with st.expander("Schema retrieval and question plan"):
        st.json({"schema_selection": result.get("schema_selection"),
                 "question_plan": result.get("question_plan"),
                 "semantic_intent": result.get("semantic_intent"),
                 "grounding": result.get("grounding"),
                 "evidence_packet_identity": result.get("evidence_packet_identity"),
                 "schema_format": result.get("schema_format"),
                 "value_profile_identity": result.get("value_profile_identity"),
                 "retrieval_seconds": result.get("retrieval_seconds"),
                 "grounding_seconds": result.get("grounding_seconds"),
                 "generation_seconds": result.get("generation_seconds")})
