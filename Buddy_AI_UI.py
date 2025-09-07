import os
import glob
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
import time

# ---------- CONFIG ----------
PDF_PATH = r"C:\Users\Adhiraj\Downloads\BuddyAI Project\BuddyAI-Onboarding-Testing-Doc.pdf"

# Hugging Face API Token
os.environ["HF_TOKEN"] = "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"  # Replace with your actual token

# OpenAI-compatible client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# ---------- PDF Handling ----------
def extract_pdf_text_chunks(pdf_path, chunk_size=1000):
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
        st.error(f"⚠️ Error reading PDF {pdf_path}: {e}")
        return []

def load_pdfs_from_folder(folder_path, chunk_size=1000):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = extract_pdf_text_chunks(pdf_file, chunk_size)
        all_chunks.extend(chunks)
    return all_chunks

# ---------- Embedding Model ----------
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    return tokenizer, model

tokenizer, model = load_embedding_model()

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
        return f"⚠️ Error from AI model: {e}"

# ---------- Custom Loader ----------
def show_loader():
    loader_html = """
    <div style="
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-color: rgba(0,0,0,0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;">
        <div style="font-size: 60px; text-align: center; animation: pulse 1.5s infinite;">
            🤖 ⚙️ 💡
            <div style="font-size:20px; color:white;">BuddyAI is thinking...</div>
        </div>
    </div>

    <style>
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    </style>
    """
    return loader_html

# ---------- Streamlit UI ----------
st.set_page_config(page_title="BuddyAI Onboarding Assistant", page_icon="🤖", layout="wide")

st.title("🤖 BuddyAI - Your Friendly Onboarding Companion 🚀")
st.markdown("Welcome! 🎉 I’ll help you understand project onboarding documents with ease.")

# Auto-load PDF
if not os.path.exists(PDF_PATH):
    st.error(f"📂 The path **{PDF_PATH}** does not exist. Please update the PDF_PATH variable.")
else:
    if os.path.isfile(PDF_PATH):
        onboarding_chunks = extract_pdf_text_chunks(PDF_PATH, chunk_size=1000)
    elif os.path.isdir(PDF_PATH):
        onboarding_chunks = load_pdfs_from_folder(PDF_PATH, chunk_size=1000)
    else:
        onboarding_chunks = []

    if onboarding_chunks:
        embeddings = [embed_text(chunk) for chunk in onboarding_chunks]
        st.success("✨ All set! You can now ask me questions below:")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(msg)
            else:
                st.chat_message("assistant").write(msg)

        # Chat input
        user_question = st.chat_input("💡 Ask me anything about the onboarding document:")

        if user_question:
            # Append user question
            st.session_state.chat_history.append(("user", user_question))
            st.chat_message("user").write(user_question)

            # Show loader
            loader_placeholder = st.empty()
            loader_placeholder.markdown(show_loader(), unsafe_allow_html=True)

            # Find context
            relevant_chunks = find_similar_chunks(user_question, embeddings, onboarding_chunks, top_k=4)
            context_text = "\n".join(relevant_chunks)

            prompt = (
                "You are BuddyAI, an onboarding assistant helping new project members. "
                "Answer ONLY based on the following information:\n\n"
                f"{context_text}\n\n"
                f"Question: {user_question}\n\n"
                "Provide a clear, helpful, and friendly answer."
            )

            # Simulate thinking delay
            time.sleep(2)

            # Get answer
            answer = ask_llama3_with_context(prompt)

            # Remove loader
            loader_placeholder.empty()

            # Append assistant answer
            st.session_state.chat_history.append(("assistant", answer))
            st.chat_message("assistant").write(answer)

    else:
        st.error("⚠️ No content found in the PDF(s).")
