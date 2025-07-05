import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import faiss
from utils import model
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("./final_project/iitk_cleaned_data.csv")
    nltk.download('punkt_tab')
    
    df = df.dropna(subset=["description"])  
    df.reset_index(drop=True, inplace=True)

    def chunk_by_sentences(text, chunk_size=30, overlap=1):
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = " ".join(sentences[i:i + chunk_size])
            if len(chunk.split()) > 10:  
                chunks.append(chunk)
        return chunks


    all_chunks = []
    for i, row in df.iterrows():
        desc = row['description']
        sentence_chunks = chunk_by_sentences(desc, chunk_size=30, overlap=1)
        all_chunks.extend(sentence_chunks)
    
    return all_chunks

@st.cache_resource
def initialize_resources(all_chunks):
    embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cpu')
    corpus_embeddings = embedder.encode(all_chunks, convert_to_tensor=True)
    corpus_embeddings_np = corpus_embeddings.cpu().detach().numpy()

    # Build FAISS index
    dimension = corpus_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings_np)

    return embedder, index

# Cache the preprocessing step
all_chunks = load_and_process_data()
embedder, index = initialize_resources(all_chunks)


def get_top_k_chunks(question, k=3):
    question_embedding = embedder.encode([question])
    D, I = index.search(np.array(question_embedding), k)
    return [all_chunks[i] for i in I[0]]


def answer_question(question):
    context_chunks = get_top_k_chunks(question, k=3)
    answers = []
    for context in context_chunks:
        result = model(question=question, context=context)
        answers.append(result)
    return answers



st.set_page_config(page_title="IITK QA Chatbot", page_icon="🤖")

st.title("IITK QA Chatbot")
st.markdown("Ask me anything about IIT Kanpur academic departments!")

# Text input box
user_question = st.text_input("Enter your question:")

if user_question:
    answers = answer_question(user_question)
    st.markdown("### Answers")
    for i, ans in enumerate(answers):
        st.write(f"**[Context {i+1}]** {ans['answer']} (Score: {ans['score']:.2f})")