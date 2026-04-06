import os
from dotenv import load_dotenv
from langsmith import Client
from agent import run_agent

load_dotenv()

client = Client()

DATASET_NAME = "DocuMind-GW-Eval"

qa_pairs = [
    {
        "query": "What is the main topic of the documents?",
        "answer": "Applications of machine learning techniques to gravitational wave detection and data analysis using LIGO and Virgo detector data."
    },
    {
        "query": "What ML techniques are used for glitch classification in gravitational wave detectors?",
        "answer": "Deep Convolutional Neural Networks (CNNs), Random Forests, Genetic Programming, and unsupervised ML methods are used for glitch classification."
    },
    {
        "query": "What is the mlgw model?",
        "answer": "mlgw is a machine learning model that generates time-domain gravitational waveforms from binary black hole mergers using PCA for dimensionality reduction and Mixture of Experts regression."
    },
    {
        "query": "Can CNNs alone be used to claim a statistically significant gravitational wave detection?",
        "answer": "No. CNNs cannot assign individual statistical significance to detections — they can only be used as trigger generators for follow-up analysis."
    },
    {
        "query": "What is the GravitySpy project?",
        "answer": "GravitySpy is a citizen science project built on the Zooniverse platform that combines citizen science and ML to classify glitches in gravitational wave data."
    },
    {
        "query": "What speedup does mlgw provide over TEOBResumS?",
        "answer": "mlgw provides a median speedup of about 44x over TEOBResumS at a starting frequency of 5Hz."
    },
    {
        "query": "What is DeepClean?",
        "answer": "DeepClean is a deep learning method used to subtract non-linear noise couplings from LIGO detector output by learning transfer functions using environmental and control data streams."
    },
    {
        "query": "What detection ratio did the CNN model achieve in the Gebhard et al. paper?",
        "answer": "The CNN model achieved approximately 89% detection ratio while producing a false positive about once every 19.5 minutes on the test set."
    },
    {
        "query": "What dimensionality reduction technique does mlgw use?",
        "answer": "mlgw uses Principal Component Analysis (PCA) to reduce the high-dimensional waveform to a low-dimensional representation before regression."
    },
    {
        "query": "What is iDQ?",
        "answer": "iDQ is a real-time glitch detection pipeline that uses supervised learning on auxiliary channel features to produce probabilistic statements about the presence of glitches in LIGO data."
    }
]

def create_dataset():
    """Create dataset in LangSmith if it doesn't exist."""
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME in existing:
        print(f"Dataset '{DATASET_NAME}' already exists. Skipping creation.")
        return client.read_dataset(dataset_name=DATASET_NAME)

    print(f"Creating dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for DocuMind RAG agent on gravitational wave papers."
    )

    client.create_examples(
        inputs=[{"query": pair["query"]} for pair in qa_pairs],
        outputs=[{"answer": pair["answer"]} for pair in qa_pairs],
        dataset_id=dataset.id
    )

    print(f"Created {len(qa_pairs)} examples in dataset.")
    return dataset

import uuid

def predict(inputs: dict) -> dict:
    query = inputs["query"]
    answer = run_agent(query, thread_id=str(uuid.uuid4()))  # ← unique thread per question
    return {"answer": answer}

def run_evaluation():
    """Run evaluation and print results."""
    print("Running evaluation...")

    from langsmith.evaluation import evaluate

    results = evaluate(
        predict,
        data=DATASET_NAME,
        experiment_prefix="documind-eval",
        metadata={"version": "1.0", "model": "llama-3.1-8b-instant"}
    )

    print("\nEvaluation complete!")
    print(f"View results at: https://smith.langchain.com")
    print(f"Project: DocuMind")
    print(f"Dataset: {DATASET_NAME}")

    return results

if __name__ == "__main__":
    print("=" * 50)
    print("DocuMind Evaluation Pipeline")
    print("=" * 50)

    create_dataset()

    results = run_evaluation()

    print("\nDone! Check LangSmith for detailed results.")