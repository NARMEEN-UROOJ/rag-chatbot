import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Initialize the FAISS vector store
db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), index_name="index")

# Load a free HuggingFace LLM (You must set your own Hugging Face API key)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature":0.5, "max_length":256}
)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
  retriever = db.as_retriever(search_kwargs={"k": 3})

)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("💬 Custom AI Chatbot")

user_input = st.text_input("Ask a question about Parallel & Distributed Computing:")

if user_input:
    result = qa_chain.run(user_input)
    st.markdown("### 🤖 Answer:")
    st
