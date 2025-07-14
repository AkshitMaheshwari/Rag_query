import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
import time
load_dotenv()
# os.environ['NVIDIA_NIM_API_KEY'] = os.getenv('NVIDIA_NIM_API_KEY')
# print("NVIDIA NIM API Key:", os.getenv('NVIDIA_NIM_API_KEY'))
llm = ChatNVIDIA(model="meta/llama3-70b-instruct",api_key="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")

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(api_key="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")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap = 50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(documents = st.session_state.final_documents, embedding = st.session_state.embeddings)

st.title("NVIDIA NIM AI Chatbot")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the context provided below. 
Please provide the most accurate response based on the question
If the context does not provide enough information, answer with "I don't know". 
<context>
{context}
<context>
Quesitons: {input}
"""
)

input_prompt = st.text_input("Enter your Question from the documents")

if st.button("document embeddings"):
    vector_embeddings()
    st.success("Document embeddings created successfully!")

if input_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':input_prompt})
    print("response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------")

