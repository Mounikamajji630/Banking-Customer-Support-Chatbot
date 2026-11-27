import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Groq AI model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Chatbot interface
st.title("Banking Customer Support Chatbot")
st.write("Ask me questions about banking services, loans, credit cards and charges.")

question = st.chat_input("Ask your banking question...")

if question:
    st.chat_message("user").write(question)

    # Retrieve relevant FAQ
    results = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    You are a banking customer support assistant.

    Answer ONLY using the provided banking information.
    Do not invent information.
    If the answer is not available, say:
    "Sorry, this information is not available in my knowledge base."

    Banking Information:
    {context}

    Customer Question:
    {question}
    """

    response = llm.invoke(prompt)

    st.chat_message("assistant").write(response.content)