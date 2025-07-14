# 🔍 RAG-Based Conversational Chatbot with Hybrid Search (FAISS + PDF Querying)

This is a **Retrieval-Augmented Generation (RAG)** based chatbot built using **FAISS** as the vector database, capable of **chatting with PDF documents**. The system performs **hybrid search (semantic + keyword matching)** to provide accurate, context-rich answers from uploaded PDFs.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing**  
  Upload any PDF file; the content is parsed and embedded using HuggingFace embeddings.

- 🔎 **Hybrid Search (Semantic + Keyword)**  
  Performs a combination of dense vector search and traditional keyword matching.

- 🧠 **Conversational Memory**  
  Keeps track of the ongoing conversation with context awareness.

- ⚙️ **FAISS Vector Store**  
  Uses FAISS for blazing-fast local vector similarity search.

- 🗣️ **LLM-Powered Responses**  
  Uses OpenAI (or Groq/Mistral/Ollama) to generate intelligent responses.

- Also Tested Pdf data retrieval with:
  - ✅ **AstraDB** (Cassandra-based cloud vector DB)
  - ✅ **NVIDIA NIM (NVIDIA Inference Microservices)** for enterprise-grade inference

---
