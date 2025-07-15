# 🔍 RAG-Based Conversational Chatbot with Hybrid Search (FAISS + PDF Querying)

This is a **Retrieval-Augmented Generation (RAG)** based chatbot built using **LangChain**, **FAISS**, and **HuggingFace Embeddings**, capable of **chatting with PDF documents**. It performs **hybrid search** (semantic + keyword matching) and uses an LLM backend like OpenAI, Groq, Mistral, or Ollama to generate accurate, context-aware responses.

👉 **[🚀 Try the App Live](https://ragquery-cwhb9ynpcverkavpuppmcr.streamlit.app/)**

---

## 🚀 Features

- 📄 **PDF Upload & Parsing**  
  Upload any PDF; it’s parsed, chunked, and embedded using HuggingFace models.

- 🔎 **Hybrid Search**  
  Combines dense vector (semantic) and keyword-based retrieval.

- 🧠 **Conversational Memory**  
  Context-aware chat that remembers prior conversation history.

- ⚙️ **FAISS Vector Store**  
  Fast local vector similarity search using FAISS.

- 🗣️ **LLM-Powered Answers**  
  Works with OpenAI, Groq (LLaMA 3), Ollama, and Mistral models.

- ✅ **Tested Integrations**  
  - **AstraDB** (Cassandra-based cloud vector DB)  
  - **NVIDIA NIM** (Enterprise-grade inference APIs)

---

## 🛠️ Tech Stack

- `LangChain`
- `Streamlit`
- `FAISS`
- `HuggingFace Embeddings`
- `Groq / OpenAI / Mistral / Ollama`
- `PyPDFLoader` for document parsing

---
