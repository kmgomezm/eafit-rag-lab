import streamlit as st
import time
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# Streamlit UI
# -------------------------

st.title("RAG vs LLM Playground")

st.sidebar.header("Hyperparameters")

model_name = st.sidebar.selectbox(
    "Model",
    ["llama3-70b-8192", "mixtral-8x7b-32768"]
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.2
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    100,
    2000,
    500
)

top_k = st.sidebar.slider(
    "Top-K",
    1,
    10,
    3
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

question = st.text_input("Ask a question about the document")

# -------------------------
# Load LLM
# -------------------------

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
)

# -------------------------
# Extract text from PDF
# -------------------------

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# -------------------------
# Create Vector Store
# -------------------------

def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)

    return db, embeddings, chunks


# -------------------------
# Cosine similarity metric
# -------------------------

def cosine_score(question, chunk, embeddings):

    q_emb = embeddings.embed_query(question)
    c_emb = embeddings.embed_query(chunk)

    score = cosine_similarity(
        [q_emb],
        [c_emb]
    )[0][0]

    return score


# -------------------------
# Run pipeline
# -------------------------

if uploaded_file and question:

    text = read_pdf(uploaded_file)

    db, embeddings, chunks = create_vectorstore(text)

    retriever = db.as_retriever(
        search_kwargs={"k": top_k}
    )

    docs = retriever.get_relevant_documents(question)

    context = "\n".join([d.page_content for d in docs])

    col1, col2, col3 = st.columns(3)

    # -------------------------
    # LLM Simple
    # -------------------------

    with col1:

        st.subheader("LLM Simple")

        start = time.time()

        response = llm.invoke(question)

        latency = time.time() - start

        st.write(response.content)
        st.caption(f"Response time: {latency:.2f}s")

    # -------------------------
    # RAG Standard
    # -------------------------

    with col2:

        st.subheader("RAG Standard")

        prompt = f"""
        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}
        """

        start = time.time()

        response = llm.invoke(prompt)

        latency = time.time() - start

        st.write(response.content)
        st.caption(f"Response time: {latency:.2f}s")

    # -------------------------
    # RAG Optimized
    # -------------------------

    with col3:

        st.subheader("RAG Optimized")

        system_prompt = """
        Answer only using the provided context.
        If the answer is not in the context say "I don't know".
        Do not invent information.
        """

        prompt = f"""
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        """

        start = time.time()

        response = llm.invoke(prompt)

        latency = time.time() - start

        st.write(response.content)
        st.caption(f"Response time: {latency:.2f}s")

        # similarity metric

        scores = []

        for doc in docs:
            score = cosine_score(
                question,
                doc.page_content,
                embeddings
            )
            scores.append(score)

        if scores:
            st.caption(
                f"Cosine similarity (avg): {np.mean(scores):.3f}"
            )