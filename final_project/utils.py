from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
