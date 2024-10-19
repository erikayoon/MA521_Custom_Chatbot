import hashlib
import os
import pickle

import faiss
import numpy as np
import openai
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings

# Access the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings (using OpenAI API)
embedding_model = OpenAIEmbeddings()


# Function to hash a chunk of text for caching
def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# Function to cache and load embeddings to avoid recomputation
def cache_embedding(text_chunk):
    cache_file = f"embedding_cache/{hash_text(text_chunk)}.pkl"

    # If embedding is already cached, load it
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Otherwise, compute and cache the embedding
    embedding = embedding_model.embed_documents([text_chunk])[0]
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding, f)
    return embedding


# Function to extract text from a PDF file in chunks for efficiency
def extract_text_from_pdf(pdf_path, page_limit=None):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        total_pages = len(reader.pages)
        if page_limit:
            total_pages = min(total_pages, page_limit)
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# Chunk text dynamically to avoid processing the entire document at once
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Vectorize each chunk using cached embeddings
def vectorize_text_chunks(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        embedding = cache_embedding(chunk)
        embeddings.append(embedding)
    return embeddings


# Create a FAISS index for fast retrieval
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index


# Retrieve relevant chunks using FAISS index
def retrieve_relevant_chunks(question, index, text_chunks, top_k=5):
    question_embedding = embedding_model.embed_query(question)
    distances, indices = index.search(
        np.array([question_embedding]).astype('float32'), top_k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks


# Generate a response using OpenAI API and the retrieved relevant chunks
def generate_response(question, index, text_chunks):
    relevant_chunks = retrieve_relevant_chunks(question, index, text_chunks)
    context = "\n".join(relevant_chunks)

    description = """
    You are an expert assistant specializing in guiding users through the patent application process. Your role is to provide clear, accurate, and concise information about various stages of patent applications, from determining patent eligibility, filing a patent, and understanding patent types, to explaining application fees, timelines, and legal requirements. You help users navigate the complex patent process with clarity and professionalism.
    """

    instructions = """
    Respond only to questions related to the patent application process, patent eligibility, and patent-related legal requirements. Avoid engaging in topics outside of the patent domain, and always ensure your responses are informative, clear, and respectful of legal boundaries. You are not a replacement for a patent attorney, so always advise users to seek formal legal assistance for detailed advice or legal concerns. When relevant, provide links to official patent resources or citation of key regulations (such as the USPTO guidelines).
    """

    # Call OpenAI API to generate a response using gpt-3.5-turbo or gpt-4
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with "gpt-4" if needed
        messages=[{
            "role": "system",
            "content": f"{description} {instructions}"
        }, {
            "role":
            "user",
            "content":
            f"Here is some information relevant to the question:\n{context}\n\nQuestion: {question}\nAnswer:"
        }],
        max_tokens=1000,  # Adjust token limit if needed
        temperature=0.7)

    return response['choices'][0]['message']['content'].strip()


# Main chatbot loop with lazy PDF processing
if __name__ == "__main__":
    pdf_path = "patent_basics.pdf"  # Replace with your PDF path
    cache_dir = "embedding_cache"

    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Process the PDF in chunks, with lazy loading to handle large files
    print("Processing PDF...")
    pdf_text = extract_text_from_pdf(
        pdf_path,
        page_limit=500)  # You can modify page limit based on your needs
    text_chunks = chunk_text(pdf_text)

    print(f"Processing {len(text_chunks)} chunks...")
    embeddings = vectorize_text_chunks(text_chunks)
    index = create_faiss_index(embeddings)

    print("Welcome to the RAG chatbot. Ask me anything!")

    # Start chatbot loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Generate chatbot response
        response = generate_response(user_input, index, text_chunks)
        print(f"Bot: {response}")
