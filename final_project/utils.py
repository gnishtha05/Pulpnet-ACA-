from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model () :
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()