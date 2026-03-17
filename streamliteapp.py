import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ---------------------------
# 🔑 API KEY (USE ENV VARIABLE)
# ---------------------------
os.environ["OPENAI_API_KEY"] = "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"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄 Chat with Your Documents (Chroma RAG)")

# ---------------------------
# Session State
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------------------
# Load Documents
# ---------------------------
def load_documents(uploaded_files):
    docs = []

    for file in uploaded_files:
        file_path = f"./temp_{file.name}"

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue

        docs.extend(loader.load())

    return docs

# ---------------------------
# Process Documents
# ---------------------------
def process_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

# ---------------------------
# Load Existing DB
# ---------------------------
def load_existing_db():
    if os.path.exists("./chroma_db"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    return None

# Load DB if exists
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_existing_db()

# ---------------------------
# Upload Files
# ---------------------------
uploaded_files = st.file_uploader(
    "📂 Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🔄 Process Documents"):
    with st.spinner("Processing..."):
        docs = load_documents(uploaded_files)
        st.session_state.vectorstore = process_documents(docs)

    st.success("✅ Done!")

# ---------------------------
# Chat
# ---------------------------
if st.session_state.vectorstore:

    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    query = st.chat_input("Ask something...")

    if query:
        result = qa.invoke({"query": query})
        answer = result["result"]

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", answer))

# ---------------------------
# Display Chat
# ---------------------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
