# # setup_data.py
# import pyreadstat
# from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

# # Step 1: Convert SAS files to CSV
# def convert_sas_to_csv(sas_file_path, csv_file_path):
#     df, meta = pyreadstat.read_sas7bdat(sas_file_path)
#     df.to_csv(csv_file_path, index=False)
#     print(f"Converted {sas_file_path} to {csv_file_path}")

# # Convert ADAE and ADSL SAS files to CSV
# convert_sas_to_csv('data/adae.sas7bdat', 'data/adae.csv')
# convert_sas_to_csv('data/adsl.sas7bdat', 'data/adsl.csv')

# # Step 2: Build LlamaIndex for each CSV file
# def build_index(csv_file_path, index_file_path):
#     data = SimpleDirectoryReader(input_path=csv_file_path).load_data()
#     index = GPTVectorStoreIndex.from_documents(data)
#     index.save_to_disk(index_file_path)
#     print(f"Created index at {index_file_path}")

# # Create indices for ADAE and ADSL datasets
# build_index('data/adae.csv', 'index/adae_index.json')
# build_index('data/adsl.csv', 'index/adsl_index.json')

# # setup_data.py
# import pyreadstat
# import pandas as pd
# from llama_index import GPTVectorStoreIndex, Document

# # Step 1: Convert SAS files to CSV
# def convert_sas_to_csv(sas_file_path, csv_file_path):
#     df, meta = pyreadstat.read_sas7bdat(sas_file_path)
#     df.to_csv(csv_file_path, index=False)
#     print(f"Converted {sas_file_path} to {csv_file_path}")

# # Convert ADAE and ADSL SAS files to CSV
# convert_sas_to_csv('data/adae.sas7bdat', 'data/adae.csv')
# convert_sas_to_csv('data/adsl.sas7bdat', 'data/adsl.csv')

# # Step 2: Build LlamaIndex for each CSV file
# def build_index(csv_file_path, index_file_path):
#     # Load CSV data into a DataFrame
#     df = pd.read_csv(csv_file_path)
    
#     # Create Document objects from DataFrame rows
#     documents = [Document(text=row.to_string(), doc_id=str(i)) for i, row in df.iterrows()]

#     # Create an index from the documents
#     index = GPTVectorStoreIndex.from_documents(documents)
#     index.save_to_disk(index_file_path)
#     print(f"Created index at {index_file_path}")

# # Create indices for ADAE and ADSL datasets
# build_index('data/adae.csv', 'index/adae_index.json')
# build_index('data/adsl.csv', 'index/adsl_index.json')

# setup_data.py
# import pyreadstat
# import pandas as pd
# from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader  # Importing Document and VectorStoreIndex
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# # Step 1: Convert SAS files to CSV
# def convert_sas_to_csv(sas_file_path, csv_file_path):
#     df, meta = pyreadstat.read_sas7bdat(sas_file_path)
#     df.to_csv(csv_file_path, index=False)
#     print(f"Converted {sas_file_path} to {csv_file_path}")

# # Convert ADAE and ADSL SAS files to CSV
# convert_sas_to_csv('data/adae.sas7bdat', 'data/adae.csv')
# convert_sas_to_csv('data/adsl.sas7bdat', 'data/adsl.csv')

# # Step 2: Build LlamaIndex for each CSV file
# def build_index(csv_file_path, index_file_path):
#     # Load CSV data into a DataFrame
#     df = pd.read_csv(csv_file_path)
    
#     # Create Document objects from DataFrame rows
#     documents = [Document(text=row.to_string(), doc_id=str(i)) for i, row in df.iterrows()]

#     # Create an index from the documents
#     index = VectorStoreIndex(documents,embed_model='local')  # Use VectorStoreIndex for indexing
#     index.save_to_disk(index_file_path)
#     print(f"Created index at {index_file_path}")

# # Create indices for ADAE and ADSL datasets
# build_index('data/adae.csv', 'index/adae_index.json')
# build_index('data/adsl.csv', 'index/adsl_index.json')

# import pyreadstat
# import pandas as pd
# from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# # Step 1: Convert SAS files to CSV
# def convert_sas_to_csv(sas_file_path, csv_file_path):
#     df, meta = pyreadstat.read_sas7bdat(sas_file_path)
#     df.to_csv(csv_file_path, index=False)
#     print(f"Converted {sas_file_path} to {csv_file_path}")

# # Convert ADAE and ADSL SAS files to CSV
# convert_sas_to_csv('data/adae.sas7bdat', 'data/adae.csv')
# convert_sas_to_csv('data/adsl.sas7bdat', 'data/adsl.csv')

# # Step 2: Build LlamaIndex for each CSV file
# def build_index(csv_file_path, index_file_path):
#     # Use SimpleDirectoryReader to read CSV data
#     reader = SimpleDirectoryReader(input_path=csv_file_path)
#     documents = reader.load_data()  # This will load the CSV as Document objects

#     # Use a local embedding model
#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Specify your local model here

#     # Create an index from the documents
#     index = VectorStoreIndex(documents, embed_model=embed_model)  # Use VectorStoreIndex for indexing with local model
#     index.save_to_disk(index_file_path)
#     print(f"Created index at {index_file_path}")

# # Create indices for ADAE and ADSL datasets
# build_index('data/adae.csv', 'index/adae_index.json')
# build_index('data/adsl.csv', 'index/adsl_index.json')

import pyreadstat
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pickle

# Step 1: Convert SAS files to CSV
def convert_sas_to_csv(sas_file_path, csv_file_path):
    df, meta = pyreadstat.read_sas7bdat(sas_file_path)
    df.to_csv(csv_file_path, index=False)
    print(f"Converted {sas_file_path} to {csv_file_path}")

# Convert ADAE and ADSL SAS files to CSV
convert_sas_to_csv('data/adae.sas7bdat', 'data/adae.csv')
convert_sas_to_csv('data/adsl.sas7bdat', 'data/adsl.csv')

# Step 2: Build LlamaIndex for each CSV file
def build_index(csv_file_path, index_file_path):
    # Load CSV data using pandas directly
    df = pd.read_csv(csv_file_path)
    
    # Create Document objects from DataFrame rows
    documents = [Document(text=row.to_string(), doc_id=str(i)) for i, row in df.iterrows()]

    # Use a local embedding model
    embed_model = HuggingFaceEmbedding(model_name="google/tapas-large-finetuned-wtq")  # Specify your local model here

    # Create an index from the documents
    index = VectorStoreIndex(documents, embed_model=embed_model)  # Use VectorStoreIndex for indexing with local model
    #index.save_to_disk(index_file_path)
    # Instead of save_to_disk, you can use the built-in method to serialize the index
    #index.serialize(index_file_path)  # Change this to the correct serialization method
     # Save the index using pickle (or use an appropriate save method if available)
    with open(index_file_path, 'wb') as f:
        pickle.dump(index, f)

    print(f"Created index at {index_file_path}")
    #print(f"Created index at {index_file_path}")


# # Create indices for ADAE and ADSL datasets
# build_index('data/adae.csv', 'index/adae_index.json')
# build_index('data/adsl.csv', 'index/adsl_index.json')

# Create indices for ADAE and ADSL datasets
build_index('data/adae.csv', 'index/adae_index.pkl')  # Using .pkl extension for pickle
build_index('data/adsl.csv', 'index/adsl_index.pkl')


