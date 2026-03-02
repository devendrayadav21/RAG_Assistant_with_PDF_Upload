import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# 1.x Compatibility Layer (Required for LangChain 1.2.10)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Standard Core & Community Imports
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Environment Setup
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize Global Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="2026 RAG Assistant", page_icon="📑")
st.title("Conversational RAG (LangChain 1.2.10)")
st.write("Upload PDFs and chat with an AI that remembers your conversation.")

# Sidebar/Header Configuration
api_key = st.text_input("Enter your Groq API key", type="password")

if api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # Create a real file path for PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Vectorization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # Chain 1: History-Aware Question Reformulation
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Using langchain_classic helper
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Chain 2: Answer Generation
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer isn't in the context, say you don't know. "
            "Limit your answer to three sentences.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Using langchain_classic helpers
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # History Management
        def get_session_history(sess_id: str) -> BaseChatMessageHistory:
            if sess_id not in st.session_state.store:
                st.session_state.store[sess_id] = ChatMessageHistory()
            return st.session_state.store[sess_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # UI Chat Interface
        user_input = st.chat_input("Ask something about the PDFs...")
        
        if user_input:
            session_history = get_session_history(session_id)
            
            # Display history for UI consistency
            for msg in session_history.messages:
                st.chat_message(msg.type).write(msg.content)

            # Generate new response
            with st.chat_message("assistant"):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write(response["answer"])
else:
    st.info("Please enter your Groq API key to unlock the chat.")