from transformers import pipeline
import streamlit as st

@st.cache_resource
def qa_pipeline () :
    return pipeline("question-answering", model="deepset/roberta-base-squad2")
