import streamlit as st
import pandas as pd
from transformers import pipeline
import pyreadstat
import os

# Create a directory to store temporary files if it doesn't exist
temp_dir = r"C:\Users\TEJA9\Desktop\Q&A Project\data"
os.makedirs(temp_dir, exist_ok=True)

# Initialize the TAPAS model
model_name = "google/tapas-large-finetuned-wtq"
qa_pipeline = pipeline("table-question-answering", model=model_name)

# Streamlit Chatbot UI
st.title("Clinical Trial Data Chatbot")
st.write("Upload your ADAE (Adverse Events) or ADSL (Subject-Level) dataset and ask questions!")

# Initialize session state for conversation history and uploaded datasets
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = None
if 'current_df_name' not in st.session_state:
    st.session_state['current_df_name'] = None

# File upload for dataset
uploaded_file = st.file_uploader("Upload Dataset (SAS .sas7bdat)", type=["sas7bdat"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    uploaded_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read the uploaded file
    st.session_state['current_df'], meta = pyreadstat.read_sas7bdat(uploaded_file_path)
    st.session_state['current_df_name'] = uploaded_file.name
    st.success(f"{uploaded_file.name} loaded successfully!")
    st.write(st.session_state['current_df'].head())  # Display the first few rows of the dataset

# User input for questions
user_input = st.text_input("You: ", "Type your question here...")

# Process the user's query
if user_input:
    if st.session_state['current_df'] is not None:
        # Convert the DataFrame to a list of dictionaries for the table
        table_data = st.session_state['current_df'].astype(str).to_dict(orient='records')
        
        # Prepare the input for the model as required
        inputs = {
            "table": table_data,
            "query": user_input
        }
        
        # Get the answer from the model
        try:
            answer = qa_pipeline(**inputs)  # Unpack inputs using **
            if isinstance(answer, list) and len(answer) > 0:
                response = answer[0]['answer']
            else:
                response = "No answer found. Please try asking a different question."
        except ValueError as e:
            response = f"Error processing the query: {e}"
        except Exception as e:
            response = f"An unexpected error occurred: {e}"
        
        st.session_state['history'].append((user_input, response))
    else:
        st.warning("Please upload a dataset before asking questions.")

# Display conversation history
if st.session_state['history']:
    for user_query, bot_response in st.session_state['history']:
        st.write(f"You: {user_query}")
        st.write(f"Bot: {bot_response}")
