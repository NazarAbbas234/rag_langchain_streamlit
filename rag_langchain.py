import os
import argparse
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def load_documents(data_path):
    """Load documents from a single .docx file or a directory of .docx files."""
    if os.path.isfile(data_path) and data_path.endswith(".docx"):
        print(f"📄 Loading single DOCX file: {data_path}")
        loader = Docx2txtLoader(data_path)
        return loader.load()
    elif os.path.isdir(data_path):
        print(f"📂 Loading all DOCX files from folder: {data_path}")
        loader = DirectoryLoader(
            data_path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
        )
        return loader.load()
    else:
        print("❌ Invalid data path. Please provide a .docx file or a folder containing .docx files.")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to .docx file or folder containing .docx files")
    parser.add_argument("--persist_dir", type=str, required=True, help="Directory to store/load FAISS index")
    args = parser.parse_args()

    # Initialize embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(args.persist_dir):
        print(f"📦 Loading existing FAISS index from {args.persist_dir} ...")
        vectorstore = FAISS.load_local(args.persist_dir, embedding, allow_dangerous_deserialization=True)
    else:
        print("📥 Loading documents...")
        documents = load_documents(args.data_path)

        if not documents:
            print("❌ No documents found. Exiting.")
            return

        print(f"✅ Loaded {len(documents)} documents. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        print("⚡ Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(args.persist_dir)
        print(f"💾 Vectorstore saved at {args.persist_dir}")

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define a custom prompt
    prompt_template = """
    You are an assistant answering questions based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer clearly and concisely:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Connect to local Ollama LLM (llama3)
    llm = ChatOpenAI(
        model="llama3",
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # placeholder, not used by Ollama
    )

    # Build RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    print("\n🚀 RAG system ready! Ask questions about your documents.\n")
    while True:
        query = input("❓ Question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting RAG.")
            break
        result = qa_chain({"query": query})
        print("\n💡 Answer:", result["result"])
        print("📚 Sources:", [doc.metadata.get("source", "unknown") for doc in result["source_documents"]])
        print("-" * 80)


if __name__ == "__main__":
    main()
