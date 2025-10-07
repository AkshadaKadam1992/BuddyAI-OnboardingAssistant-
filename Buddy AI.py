import os
import glob
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from numpy.linalg import norm
from openai import OpenAI

# Set your Hugging Face API token here
os.environ["HF_TOKEN"] = "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"  # Replace with your actual token

# Initialize OpenAI-compatible client for Hugging Face Meta Llama 3.1 inference
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def extract_pdf_text_chunks(pdf_path, chunk_size=1000):
    """Extract text from PDF and split into chunks"""
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []

def load_pdfs_from_folder(folder_path, chunk_size=1000):
    """Load all PDF files from a folder and extract text chunks"""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in folder: {folder_path}")
        return []
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = extract_pdf_text_chunks(pdf_file, chunk_size)
        all_chunks.extend(chunks)
    return all_chunks

print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
print("Embedding model loaded successfully!")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def find_similar_chunks(query, embeddings, chunks, top_k=4):
    query_embedding = embed_text(query)
    similarities = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(embeddings)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in similarities[:top_k]]

def ask_llama3_with_context(prompt):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error from AI model: {e}"

def main():
    print("=" * 60)
    print("Welcome to Project BuddyAI - Your Intelligent Onboarding Companion!")
    print("=" * 60)

    # Hardcode your path here - can be a folder or single pdf
    path = r"C:\Users\Adhiraj\Downloads\BuddyAI Project\BuddyAI-Onboarding-Testing-Doc.pdf"  # Or folder path

    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        return

    if os.path.isfile(path):
        print(f"Processing single PDF file: {path}")
        onboarding_chunks = extract_pdf_text_chunks(path, chunk_size=1000)
    elif os.path.isdir(path):
        print(f"Processing all PDFs in folder: {path}")
        onboarding_chunks = load_pdfs_from_folder(path, chunk_size=1000)
    else:
        print(f"Error: Path '{path}' is neither a file nor a directory.")
        return

    if not onboarding_chunks:
        print("No content found in the specified PDF(s). Exiting.")
        return

    print(f"\nGenerating embeddings for {len(onboarding_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(onboarding_chunks):
        if i % 10 == 0:
            print(f"Embedding chunk {i+1}/{len(onboarding_chunks)}...")
        embeddings.append(embed_text(chunk))

    print("\n" + "=" * 60)
    print("BuddyAI is ready to answer your onboarding questions!")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using BuddyAI! Happy onboarding! 🚀")
            break
        if not user_question:
            print("Please enter a valid question.")
            continue

        print("\nSearching for relevant information...")
        relevant_chunks = find_similar_chunks(user_question, embeddings, onboarding_chunks, top_k=4)
        context_text = "\n".join(relevant_chunks)
        prompt = (
            "You are BuddyAI, an onboarding assistant helping new project members. "
            "Answer ONLY based on the following information about customer ethics, compliance, "
            "coding standards, technical background, and business processes:\n\n"
            f"{context_text}\n\n"
            f"Question: {user_question}\n\n"
            "Provide a helpful and accurate answer based on the provided information."
        )

        print("Generating answer...")
        answer = ask_llama3_with_context(prompt)

        print("\n" + "-" * 50)
        print("BuddyAI Answer:")
        print("-" * 50)
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
