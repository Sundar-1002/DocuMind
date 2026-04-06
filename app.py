import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/graphql"

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

st.title("📄 DocuMind")
st.caption("Upload your PDFs and ask questions about them.")


# --- Helper: send GraphQL mutation ---
def ask_question(question: str) -> str:
    query = """
        mutation($question: MessageInput!) {
            ask(question: $question) {
                role
                content
            }
        }
    """
    response = requests.post(
        API_URL,
        json={
            "query": query,
            "variables": {
                "question": {
                    "role": "user",
                    "content": question
                }
            }
        }
    )
    data = response.json()
    return data["data"]["ask"]["content"]


def upload_documents(uploaded_files) -> str:
    # multipart upload for GraphQL file upload spec
    operations = json.dumps({
        "query": """
            mutation($files: [Upload!]!) {
                uploadDocuments(files: $files)
            }
        """,
        "variables": {
            "files": [None] * len(uploaded_files)
        }
    })

    # map each file to its position in the variables array
    file_map = json.dumps({
        str(i): [f"variables.files.{i}"]
        for i in range(len(uploaded_files))
    })

    # build multipart form
    files = {"operations": (None, operations), "map": (None, file_map)}
    for i, f in enumerate(uploaded_files):
        files[str(i)] = (f.name, f.getbuffer(), "application/pdf")

    response = requests.post(API_URL, files=files)
    data = response.json()
    return data["data"]["uploadDocuments"]


# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Upload & Ingest", use_container_width=True):
        with st.spinner("Uploading and ingesting documents..."):
            message = upload_documents(uploaded_files)
        st.success(message)


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []


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
            answer = ask_question(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})