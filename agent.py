import langchain
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
CHROMA_DB = "chroma_db/"

class AgentState(TypedDict):
    query: List[Union[SystemMessage, HumanMessage, AIMessage]]
    documents: List[Document]
    answer:str

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings
)
llm = ChatGroq(model="llama-3.1-8b-instant")
def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieves relevant documents based on the query and updates the state with those documents."""
    print("Retriving Relavant Documents for the Query...")
    docs = vectorstore.similarity_search(state["query"])
    return {"documents": docs}

def respond(state: AgentState) -> AgentState:
    """Generates a response based on the query and retrieved documents, and updates the state with the answer."""
    print("Generating Response...")

    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    source_info = list(set([doc.metadata.get("source", "Unknown") for doc in state["documents"]]))

    system_message = SystemMessage(content=f"Answer the question based only on the context provided below. If the answer is not in the context, say 'I could not find this in the provided documents.'.\n\nContext:\n{context}. Always ask user if they want to know more until they say no.")
    human_message =HumanMessage(content=state["query"])

    state["query"] = [system_message, human_message]

    response = llm.invoke(state["query"])
    final_answer = f"{response.content}\n\nSources: {', '.join(source_info)}"
    return {"answer": final_answer}

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("respond", respond)
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "respond")
    graph.add_edge("respond", END)
    return graph.compile()

agent = build_graph()

def run_agent(query: str) -> str:
    result = agent.invoke({"query": query})
    return result["answer"]

if __name__ == "__main__":
    print("Welcome to DocuMind! Ask any question related to the documents you have ingested.\nType 'exit' to quit.")
    while True:
        user_query = input("Your Question: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        answer = run_agent(user_query)
        print(f"Answer: {answer}\n")