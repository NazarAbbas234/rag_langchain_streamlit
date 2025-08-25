import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
import os

# Paths
DATA_PATH = "data"
PERSIST_DIR = "faiss_index"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index if exists, otherwise create it
if os.path.exists(PERSIST_DIR):
    vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    loader = UnstructuredFileLoader(os.path.join(DATA_PATH, "handbook.docx"))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(PERSIST_DIR)

# Initialize LLM (Ollama must be running)
llm = Ollama(model="llama3")

# Conversational chain
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Streamlit UI
st.set_page_config(page_title="📘 Handbook Q&A", layout="centered")
st.title("📘 Handbook Chatbot")

# Session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input (with form to reset automatically)
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Ask a question about the handbook:")
    submit = st.form_submit_button("Send")

if submit and query:
    result = qa({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))

# Display conversation
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
