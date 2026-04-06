# 📄 DocuMind — AI-Powered Document Research Agent

DocuMind is a RAG-powered research agent built with LangGraph and LangChain that lets you upload PDF documents and ask questions about them — returning answers with cited source filenames, powered by Llama 3.1 and ChromaDB.

---

## 📸 Screenshots

**Chat with cited answers**
![Chat](Assets/1.png)
![Chat](Assets/2.png)

---

## 🚀 Features

- 📁 Upload multiple PDFs directly from the browser
- 🔍 Semantic search over document chunks using ChromaDB
- 🤖 LangGraph stateful agent with retriever, responder, and critic nodes
- 🔁 Self-correction loop — critic re-routes low-confidence answers back to responder
- 💬 Multi-turn conversation memory using LangGraph's `MemorySaver`
- 📄 Cited answers with source filenames
- 🖥️ Clean Streamlit UI with chat interface
- 📊 LangSmith observability and evaluation integration

---

## 🏗️ Architecture

```
User Question
      ↓
 LangGraph Agent
  ├── Retrieve Node   → ChromaDB semantic search (HuggingFace embeddings)
  ├── Respond Node    → Llama 3.1 via Groq API (cited answer)
  └── Critic Node     → Verifies answer quality and groundedness
        ↓
  Confidence HIGH → Return answer to user
  Confidence LOW  → Route back to Respond (max 2 retries with critic feedback)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph + LangChain |
| LLM | Llama 3.1 8B via Groq API |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| PDF Parsing | LangChain PyPDFDirectoryLoader |
| UI | Streamlit |
| Observability | LangSmith |
| Language | Python 3.11 |

---

## 📦 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sundar-1002/DocuMind.git
cd DocuMind
```

**2. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=DocuMind
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)
Get your free LangSmith API key at [smith.langchain.com](https://smith.langchain.com)

---

## 📁 Project Structure

```
DocuMind/
├── Assets/                        ← Screenshots and evaluation results
│   ├── 1.png
│   ├── 2.png
│   └── documind-eval-cb6ac171.csv ← LangSmith evaluation results
├── data/                          ← Place your PDF files here
├── chroma_db/                     ← Auto-generated vector store
├── ingest.py                      ← PDF loading, chunking, embedding pipeline
├── agent.py                       ← LangGraph agent graph with critic node
├── app.py                         ← Streamlit UI
├── evaluate.py                    ← LangSmith evaluation pipeline
├── requirements.txt
└── .env
```

---

## ▶️ Usage

**Step 1 — Ingest your documents**

Place PDF files in the `data/` folder, then run:

```bash
python ingest.py
```

Or use the **Ingest Documents** button directly in the UI.

**Step 2 — Run the app**

```bash
streamlit run app.py
```

**Step 3 — Ask questions**

Open `http://localhost:8501` in your browser, upload PDFs, and start chatting!

**Step 4 — Run evaluation (optional)**

```bash
python evaluate.py
```

---

## 📋 Requirements

```
langchain
langchain-community
langchain-google-genai
langchain-huggingface
langchain-chroma
langchain-groq
langchain-text-splitters
langgraph
chromadb
streamlit
pypdf
sentence-transformers
google-generativeai
python-dotenv
langsmith
```

---

## 💡 How It Works

**Ingestion Pipeline (`ingest.py`)**
PDFs are loaded page by page using `PyPDFDirectoryLoader`, split into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`, embedded using HuggingFace's `all-MiniLM-L6-v2` model, and persisted locally in ChromaDB.

**Agent Graph (`agent.py`)**
A LangGraph `StateGraph` with three nodes — `retrieve_documents` performs semantic similarity search against ChromaDB returning the top 4 relevant chunks, `respond` builds a prompt with the retrieved context and invokes Llama 3.1 via Groq to generate a cited answer, and `critic` verifies whether the answer is grounded in the retrieved documents. If confidence is LOW, the critic's feedback is passed back to the responder for a retry (max 2 retries). Conversation memory is maintained across turns using `MemorySaver`.

**Evaluation Pipeline (`evaluate.py`)**
A LangSmith evaluation pipeline that runs the agent against a dataset of 10 question-answer pairs and logs latency, token usage, and answer quality to LangSmith.

---

## 📊 Evaluation Results

Evaluated on 10 domain-specific questions from gravitational wave research papers using LangSmith. All 10 runs completed successfully.

| Question | Status | Latency |
|----------|--------|---------|
| What is the main topic of the documents? | ✅ | 12.9s |
| What ML techniques are used for glitch classification? | ✅ | 1.1s |
| What is the mlgw model? | ✅ | 0.4s |
| Can CNNs alone claim a statistically significant detection? | ✅ | 9.5s |
| What is the GravitySpy project? | ✅ | 10.6s |
| What speedup does mlgw provide over TEOBResumS? | ✅ | 0.6s |
| What is DeepClean? | ✅ | 11.5s |
| What detection ratio did the CNN model achieve? | ✅ | 0.2s |
| What dimensionality reduction technique does mlgw use? | ✅ | 0.5s |
| What is iDQ? | ✅ | 0.5s |

Full results available in `Assets/documind-eval-cb6ac171.csv`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

MIT License
