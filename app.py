import os
import streamlit as st
from ingest import ingest
from agent import run_agent

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

st.title("📄 DocuMind")
st.caption("Upload your PDFs and ask questions about them.")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join("data", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded to data/")

    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents into ChromaDB..."):
            ingest()
        st.success("Documents ingested successfully!")

# --- Session State ---                              # ← ADD THIS BLOCK
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:             # ← ADD THIS
    st.session_state.thread_id = "user_session_1"   # ← ADD THIS

# --- Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_agent(prompt, thread_id=st.session_state.thread_id)  # ← UPDATED
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})