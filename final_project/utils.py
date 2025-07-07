from transformers import pipeline, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_model () :
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model="google/flan-t5-base", tokenizer=tokenizer), tokenizer


model, tokenizer = load_model()





