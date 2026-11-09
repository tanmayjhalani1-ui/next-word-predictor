from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference import load_assets, load_metadata, predict_next_words

st.set_page_config(page_title="Next Word Predictor", page_icon="⌨️", layout="wide")
st.title("Next Word Predictor")
st.caption("An LSTM language-model experiment trained on a small CampusX DSMP FAQ-style corpus.")

@st.cache_resource
def get_assets():
    return load_assets()

metadata = load_metadata()
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("Generate suggestions")
    seed = st.text_area("Enter a sentence or phrase", placeholder="What is the course")
    count = st.slider("Number of suggestions", 1, 10, 5)
    temperature = st.slider("Sampling temperature", 0.2, 1.5, 0.8, 0.1)
    if st.button("Predict next words", type="primary"):
        if not seed.strip():
            st.warning("Enter some text first.")
        else:
            try:
                model, tokenizer, info = get_assets()
                suggestions = predict_next_words(seed.strip(), model, tokenizer, info.get("max_sequence_length", 40), count, temperature)
                if suggestions:
                    st.dataframe(pd.DataFrame(suggestions).rename(columns={"word": "Suggested word", "probability": "Model probability"}), hide_index=True, use_container_width=True)
                else:
                    st.info("No suggestions were generated.")
            except (FileNotFoundError, ValueError) as error:
                st.warning(str(error))

with right:
    st.subheader("Experiment overview")
    if metadata:
        st.metric("Selected model", metadata.get("selected_model", "—"))
        st.metric("Vocabulary size", metadata.get("vocabulary_size", "—"))
        st.dataframe(pd.DataFrame(metadata.get("candidates", [])), hide_index=True, use_container_width=True)
        st.caption(metadata.get("limitations", ""))
    else:
        st.info("Train the model before making predictions.")
        st.code("python -m src.train", language="bash")
    st.caption("Suggestions are statistical next-token predictions. They are not verified answers.")
