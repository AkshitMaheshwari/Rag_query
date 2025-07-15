import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-miniLM-L6-v2")

st.set_page_config(page_title="RAG Conversational Chatbot", layout="wide",page_icon="📄")

st.title("🤖 AI Conversational Chatbot with PDF Query")

st.write("Chat with me! You can also upload PDFs to ask questions about specific documents.")

llm = ChatOpenAI(model = "gpt-4.1",base_url = "https://models.inference.ai.azure.com",api_key = os.getenv("OPENAI_API_KEY"))


if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None


with st.sidebar:
    st.subheader("🔧 Chat Mode Selection")

    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "conversational"
    session_id = st.text_input("Session ID", value="default_session", key="session_id")
    # Mode switch buttons
    if st.session_state.chat_mode == "conversational":
        st.write("🧠 You are in **Conversational Mode**")
        if st.button("🔄 Switch to PDF Chatbot"):
            st.session_state.chat_mode = "pdf"
            st.rerun()
    else:
        st.write("📄 You are in **PDF Chatbot Mode**")
        if st.button("🔄 Switch to General Chat"):
            st.session_state.chat_mode = "conversational"
            st.session_state.pdf_processed = False
            st.session_state.conversational_chain = None
            st.rerun()

    # PDF upload UI only in PDF mode
    if st.session_state.chat_mode == "pdf":
        st.subheader("📄 Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    else:
        uploaded_files = None  

    st.subheader("🗑️ Clear Chat History")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if session_id in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        st.session_state.pdf_processed = False
        st.session_state.conversational_chain = None
        st.rerun()

    st.subheader("📊 Chat Statistics")
    if st.session_state.messages:
        st.write(f"Total messages: {len(st.session_state.messages)}")
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(f"User messages: {user_messages}")
        st.write(f"Assistant messages: {len(st.session_state.messages) - user_messages}")

if st.session_state.chat_mode == "pdf" and uploaded_files and not st.session_state.pdf_processed:
    with st.spinner("Processing PDFs..."):
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt = (
                "Given a chat history and the latest question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do not answer the question, "
                "just reformulate it if needed otherwise return as it is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Answer the question in detail using the following pieces of retrieved context. "
                "If you don't know the answer, say that you don't have enough information about this topic. "
                "Provide detailed answers for the question.\n\n{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)
            
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]
            
            st.session_state.conversational_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_message_key="answer"
            )
            
            st.session_state.pdf_processed = True
            st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully! Now I can answer questions about your documents.")
            
            for uploaded_file in uploaded_files:
                temppdf = f"./temp_{uploaded_file.name}"
                try:
                    os.remove(temppdf)
                except:
                    pass

if not st.session_state.conversational_chain:
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    from langchain_core.runnables import RunnableLambda
    
    def simple_chat_function(inputs):
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        
        messages = [
            ("system", "You are a helpful, friendly AI assistant. Have natural conversations with users.")
        ]
        
        for msg in chat_history:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                messages.append((msg.type, msg.content))
        
        messages.append(("human", user_input))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        response = llm.invoke(prompt.format_messages())
        
        return {"answer": response.content}
    
    simple_runnable = RunnableLambda(simple_chat_function)
    
    st.session_state.conversational_chain = RunnableWithMessageHistory(
        simple_runnable, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_message_key="answer"
    )

for message in st.session_state.messages:  
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Ask me anything or upload PDFs to chat about documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversational_chain.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {"session_id": session_id}
                    }, 
                )
                assistant_response = response["answer"]
                st.markdown(assistant_response)
                
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome_msg = "👋 Hi! I'm your AI assistant. You can:\n\n• Have a general conversation with me\n• Upload PDFs in the sidebar to ask questions about documents\n• Switch between chat modes anytime\n\nWhat would you like to talk about?"
        st.markdown(welcome_msg)
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

