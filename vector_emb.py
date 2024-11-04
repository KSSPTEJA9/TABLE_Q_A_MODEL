import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
import torch

def main():
    # Load your CSV file
    df = pd.read_csv('data/adae.csv')

    # Specify the columns for text data (update this list with all relevant text columns)
    #text_columns = ['AETERM', 'AEDECOD', 'AEBODSYS', 'AEHLGT', 'AEHLT', 'AELLT']  # Add more columns as needed

    text_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Combine the text from specified columns into a single string for each row
    #df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

    # Load a pre-trained model
    model = SentenceTransformer('deepset/all-mpnet-base-v2-table')

    # Create a directory to save indices if it doesn't exist
    os.makedirs('index', exist_ok=True)

    # Dictionary to hold FAISS indices for each column
    indices = {}

 # Collect tables for TAPAS
    tables = []

    for column in text_columns:
        # Define index file path
        index_file_path = f'index/{column}_index.index'
        
        # Check if index already exists
        if os.path.exists(index_file_path):
            print(f"Loading existing index for column '{column}' from {index_file_path}")
            index = faiss.read_index(index_file_path)
        else:
            print(f"Generating new index for column '{column}'")
            # Generate embeddings for each specified column
            embeddings = model.encode(df[column].fillna('').tolist(), convert_to_tensor=True)
            embeddings_np = np.array(embeddings.tolist()).astype('float32')

            # Create a FAISS index for this column
            index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Using L2 distance

            # Add the embeddings to the index
            index.add(embeddings_np)

            # Save the index in the 'index' directory
            faiss.write_index(index, index_file_path)

        # Store the index in the dictionary
        indices[column] = index

    # Example query for a specific column
    query_text = "frequent AETERM?"
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding.tolist()).astype('float32').reshape(1, -1)

    # Search for the top 2 nearest neighbors in each index
    k = 2
    for column, index in indices.items():
        distances, neighbors_indices = index.search(query_embedding_np, k)

        # Create a DataFrame for the nearest neighbors and retain only the text column with its header
        nearest_neighbors_df = pd.DataFrame({
            column: df[column].iloc[neighbors_indices[0]].tolist()
        })

        # Print the results for the current column
        print(f"Nearest Neighbors for {column}:")
        print(nearest_neighbors_df)
        print("\n")

        # Collect the table for TAPAS with only the text column named after the original column
        tables.append(nearest_neighbors_df)

    # Concatenate all nearest neighbor DataFrames into a single table with unique column headers
    combined_table = pd.concat(tables, axis=1)

    print(combined_table)

    # Ensure all columns are strings, replacing NaN with empty strings
    combined_table = combined_table.fillna('').astype(str)



    # Prepare TAPAS model for question answering
    model_name = "google/tapas-base-finetuned-wtq"
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)

    # Set up the TAPAS pipeline, optionally using GPU if available
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("table-question-answering", model=model, tokenizer=tokenizer, device=device)

    # Define your query
    query = "What are the frequent AETERM?"

    # Get the answer from TAPAS
    answer = pipe(table=combined_table, query=query)

    # Print the answer in a readable format
    print("TAPAS Answer:")
    print(answer)
    # for response in answer:
    #     print(f"Answer: {response['answer']}")
    #     print(f"Confidence: {response.get('score', 'N/A')}")
    #     print("\n")

if __name__ == "__main__":
    main()