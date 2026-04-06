import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "DocuMind")
CHROMA_DB = "chroma_db/"

class AgentState(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    confidence: str
    retry_count: int
    critic_output: Union[str, None]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings
)
llm = ChatGroq(model="llama-3.1-8b-instant")

def retrieve_documents(state: AgentState) -> AgentState:
    print("Retrieving Relevant Documents for the Query...")
    docs = vectorstore.similarity_search(state["query"])
    return {"documents": docs}

def respond(state: AgentState) -> AgentState:
    print(f"Generating Response... (attempt {state.get('retry_count', 0) + 1})")

    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    source_info = list(set([doc.metadata.get("source", "Unknown") for doc in state["documents"]]))

    system_message = SystemMessage(content=f"Answer the question based only on the context provided below. If the answer is not in the context, say 'I could not find this in the provided documents.'.\n\nContext:\n{context}. Always ask user if they want to know more until they say no.")
    if state.get("critic_output") is not None:
        system_message.content += f"\n\nPrevious Critic Feedback:\n{state['critic_output']}"
    human_message = HumanMessage(content=state["query"])

    response = llm.invoke([system_message, human_message])
    source_names = [os.path.basename(s) for s in source_info]
    final_answer = f"{response.content}\n\nSources: {', '.join(source_names)}"

    return {
        "answer": final_answer,
        "retry_count": state.get("retry_count", 0) + 1
    }

def critic(state: AgentState) -> AgentState:
    print("Verifying Answer Quality...")

    context = "\n\n".join([doc.page_content for doc in state["documents"]])

    system_message = SystemMessage(content=
    """You are a strict answer verifier. Given a question, a context, and an answer — evaluate the answer.
    Respond ONLY in this exact format:
    GROUNDED: yes or no
    CONFIDENCE: HIGH or LOW
    REASON: one sentence explanation""")

    human_message = HumanMessage(content=f"""Question: {state["query"]}

    Context:
    {context}

    Answer:
    {state["answer"]}""")
    response = llm.invoke([system_message, human_message])
    response_text = response.content
    result = response_text.upper()

    is_grounded = "GROUNDED: YES" in result
    confidence = "HIGH" if "CONFIDENCE: HIGH" in result else "LOW"

    print(f"Critic verdict → Grounded: {is_grounded} | Confidence: {confidence}")

    return {
        "critic_output": response_text,
        "confidence": confidence
    }

def route_after_critic(state: AgentState) -> str:
    if state["confidence"] == "LOW" and state.get("retry_count", 0) < 2:
        print(f"Confidence LOW — retrying...")
        return "respond"
    return END

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("respond", respond)
    graph.add_node("critic", critic)
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "respond")
    graph.add_edge("respond", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "respond": "respond",
            END: END
        }
    )
    return graph.compile(checkpointer=MemorySaver())

agent = build_graph()

def run_agent(query: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {
            "query": query,
            "retry_count": 0
        },
        config=config
    )
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