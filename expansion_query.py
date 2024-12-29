from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Loading  environment variables from .env file
load_dotenv()

# Setting environment variables
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Processing and extracting only text from the entire PDF file
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filtering the empty strings
pdf_texts = [text for text in pdf_texts if text]
# print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# )


# Importing text splitting functions from Langchain for breaking text into processable chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Splitting the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Assigning 256 tokens per chunk, making the data more manageable to process. 
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(word_wrap(token_split_texts[0]))
#print(f"\nTotal chunks: {len(token_split_texts)}")

# Instantiating the sentence transforming embedding functions
embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

# Vectorizing the processed microsoft document 
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# Add the tokenized chunks of text to the empty array of vecotrized inf
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

#count = chroma_collection.count()
#print(count)

# initial query
query = "What was the total revenue for the year?"

# get the raw results from a simple inital query
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# print out the inital results
#for document in retrieved_documents:
#    print(word_wrap(document))
#    print("\n")