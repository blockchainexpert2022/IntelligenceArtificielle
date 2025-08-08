#!pip install pypdf sentence-transformers faiss-cpu

import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import warnings

# Suppress a specific UserWarning from huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.inference._legacy")

# --- CONFIGURATION ---
PDF_PATH = "GrosFichierAvecDuCodeDeOuf.pdf"  # IMPORTANT: Change this to the path of your PDF
HF_TOKEN = "replace me"  # Recommended: Set your Hugging Face token
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Powerful open-source model
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Efficient and effective embedding model
CHUNK_SIZE = 500  # Number of characters per text chunk
CHUNK_OVERLAP = 50 # Number of characters to overlap between chunks

# --- INITIALIZE CLIENTS ---
# Initialize the LLM client from Hugging Face
# It's recommended to set the HUGGING_FACE_HUB_TOKEN environment variable
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

# Initialize the model for creating embeddings
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# --- STEP 1: LOAD AND CHUNK THE DOCUMENT ---
def load_and_chunk_pdf(file_path):
    """
    Loads text from a PDF, cleans it, and splits it into smaller chunks.
    """
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return None
        
    print(f"Loading and processing PDF: {file_path}...")
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + " "

    # Basic text cleaning
    full_text = " ".join(full_text.strip().split())
    
    # Split the text into chunks
    chunks = [
        full_text[i:i + CHUNK_SIZE]
        for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP)
    ]
    print(f"Successfully created {len(chunks)} text chunks.")
    return chunks

# --- STEP 2: EMBED CHUNKS AND CREATE A VECTOR STORE ---
def create_vector_store(chunks):
    """
    Creates vector embeddings for text chunks and stores them in a FAISS index.
    """
    print("Generating embeddings for text chunks...")
    # Generate embeddings for each chunk
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    
    # Create a FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    
    print("Vector store created successfully.")
    return index, embeddings

# --- STEP 3: RETRIEVE, AUGMENT, AND GENERATE ---
def query_rag_system(question, index, chunks):
    """
    Answers a question by retrieving relevant context and querying the LLM.
    """
    print(f"\nProcessing query: '{question}'")
    # 1. Retrieve: Find relevant document chunks
    question_embedding = embedding_model.encode([question])
    
    # Search the FAISS index for the top 3 most similar chunks
    D, I = index.search(np.array(question_embedding).astype('float32'), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    
    # 2. Augment: Create a detailed prompt for the LLM
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt_template = f"""
    Based on the following context extracted from a document, please answer the user's question.
    If the context does not contain the answer, state that you cannot find the answer in the provided document.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    print("Querying the LLM with retrieved context...")
    # 3. Generate: Get the final answer from the LLM
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt_template}],
            max_tokens=500,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while querying the LLM: {e}"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create a dummy PDF for demonstration if one doesn't exist
    if not os.path.exists(PDF_PATH):
        print(f"'{PDF_PATH}' not found. You should replace this with your own PDF.")
        print("Create a dummy PDF named 'your_large_pdf_file.pdf' to run this example.")
    else:
        # Build the RAG system
        text_chunks = load_and_chunk_pdf(PDF_PATH)
        if text_chunks:
            vector_store, _ = create_vector_store(text_chunks)
            
            # Now you can ask questions about your PDF
            user_question = "Ce document montre du code windev relatif à tout un projet. Identifie tous les codes qui écrivent sur un champ NuméroFacture." # Change this to your question
            answer = query_rag_system(user_question, vector_store, text_chunks)
            print("\n--- ANSWER ---")
            print(answer)
            print("--------------\n")

            # Example of another question
            user_question_2 = "Summarize the key findings mentioned in the introduction." # Change this to your question
            answer_2 = query_rag_system(user_question_2, vector_store, text_chunks)
            print("\n--- ANSWER ---")
            print(answer_2)
            print("--------------\n")
