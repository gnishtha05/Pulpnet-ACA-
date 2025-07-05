from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model () :
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

model = load_model()