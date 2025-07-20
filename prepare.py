import os
import json
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Setup
PDF_PATH = "test.pdf"
JSON_PATH = "data/pdf_chunks.json"
CSV_PATH = "data/test_qa.csv"
EMBEDDING_DIR = "data/embeddings/"
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Step 1: Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Step 2: Split and add metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

json_data = []
for doc in split_docs:
    json_data.append({
        "topic": "unknown",  # Placeholder
        "subtopic": "unknown",  # Placeholder
        "description": doc.page_content.strip(),
        "metadata": {
            "pdf_source": PDF_PATH,
            "page_number": doc.metadata.get("page", -1)
        }
    })

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

print(f"[✔] Extracted and saved {len(json_data)} JSON chunks.")

from llm_huggingface import HuggingFaceChatLLM
llm = HuggingFaceChatLLM()

qa_data = []
for item in json_data:
    text = item["description"]
    prompt = f"Generate a question and answer based on this text:\n\n{text}\n\nQ:"

    response = llm.invoke(prompt)

    if "A:" in response:
        question, answer = response.split("A:", 1)
    else:
        question, answer = response, "Not answered."

    qa_data.append({
        "topic": item["topic"],
        "subtopic": item["subtopic"],
        "question": question.strip().replace("\n", " "),
        "answer": answer.strip().replace("\n", " "),
        "description": item["description"],
        "pdf_source": item["metadata"]["pdf_source"],
        "page_number": item["metadata"]["page_number"]
    })
# Step 4: Save to CSV
df = pd.DataFrame(qa_data)
df.to_csv(CSV_PATH, index=False)
print(f"[✔] Saved Q&A pairs to {CSV_PATH}")

# Step 5: Embedding + FAISS
embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
questions = df["question"].tolist()
db = FAISS.from_texts(questions, embedder)
db.save_local(EMBEDDING_DIR)
print(f"[✔] FAISS index and embeddings saved to {EMBEDDING_DIR}")
