import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
import torch

def create_faiss_indices(df, text_columns, model):
    indices = {}
    os.makedirs('index', exist_ok=True)
    
    for column in text_columns:
        index_file_path = f'index/{column}_index.index'
        
        if os.path.exists(index_file_path):
            st.write(f"Loading existing index for column '{column}' from {index_file_path}")
            index = faiss.read_index(index_file_path)
        else:
            st.write(f"Generating new index for column '{column}'")
            embeddings = model.encode(df[column].fillna('').tolist(), convert_to_tensor=True)
            embeddings_np = np.array(embeddings.tolist()).astype('float32')
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)
            faiss.write_index(index, index_file_path)
        indices[column] = index
    
    return indices

def retrieve_nearest_neighbors(query_text, indices, text_columns, df, model, k=2):
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding.tolist()).astype('float32').reshape(1, -1)
    
    tables = []
    for column, index in indices.items():
        distances, neighbors_indices = index.search(query_embedding_np, k)
        nearest_neighbors_df = pd.DataFrame({
            column: df[column].iloc[neighbors_indices[0]].tolist()
        })
        tables.append(nearest_neighbors_df)
    
    combined_table = pd.concat(tables, axis=1).fillna('').astype(str)
    return combined_table

def main():
    st.title("SAS Data Question Answering App")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        embedding_model = SentenceTransformer('deepset/all-mpnet-base-v2-table')
        indices = create_faiss_indices(df, text_columns, embedding_model)

        query_text = st.text_input("Ask a question about the data:")
        if query_text:
            st.write(f"Query: {query_text}")
            
            combined_table = retrieve_nearest_neighbors(query_text, indices, text_columns, df, embedding_model, k=2)
            st.write("Combined Table for TAPAS Query:", combined_table)

            model_name = "google/tapas-base-finetuned-wtq"
            tokenizer = TapasTokenizer.from_pretrained(model_name)
            qa_model = TapasForQuestionAnswering.from_pretrained(model_name)
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("table-question-answering", model=qa_model, tokenizer=tokenizer, device=device)

            answer = pipe(table=combined_table, query=query_text)
            st.write("TAPAS Answer:", answer['answer'] if 'answer' in answer else answer)

if __name__ == "__main__":
    main()
