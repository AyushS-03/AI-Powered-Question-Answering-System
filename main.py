import os
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import faiss
import numpy as np
import torch
import openai
from transformers import BertTokenizer, BertModel

# Set your OpenAI API key
openai.api_key = 'you_api_key'

# Load your dataset
data = {
    'text': [
        "Python is a high-level, interpreted programming language.",
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "Machine learning is a field of artificial intelligence.",
        "OpenAI develops artificial intelligence technologies.",
        "My name is Suryasnato Mitra",
        "Taj Mahal is in Agra"
    ]
}
df = pd.DataFrame(data)

# Add titles to the dataset
df['title'] = df['text']  # Using text as title for simplicity

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(df)

# Tokenize the texts
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
encoded_passages = tokenizer(dataset['text'], padding=True, truncation=True, return_tensors="pt", max_length=512)

# Generate BERT embeddings for passages
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

inputs = bert_tokenizer(dataset['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
with torch.no_grad():
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

# Ensure embeddings are in float32
embeddings = embeddings.astype(np.float32)

# Add embeddings to the dataset
dataset = dataset.add_column("embeddings", embeddings.tolist())

# Save the dataset to disk
dataset_path = "custom_dataset"
dataset.save_to_disk(dataset_path)

# Initialize FAISS index
d = embeddings.shape[1]  # dimension of the input vectors
index = faiss.IndexFlatL2(d)  # FAISS index

# Add embeddings to the index
index.add(embeddings)

# Save the index to disk
index_path = "custom_index.faiss"
faiss.write_index(index, index_path)

# Load the dataset and index for the retriever
dataset = load_from_disk(dataset_path)
index = faiss.read_index(index_path)

# Set up the retriever with the custom index and dataset
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path
)

# Set up the model
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Define a function to generate answers using GPT
def answer_question_with_gpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use a supported model
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Define a function to generate answers using the RAG model (fallback)
def answer_question_with_rag(question):
    inputs = tokenizer(question, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs['input_ids'], num_return_sequences=1, num_beams=1)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Set up a simple Q&A session
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    answer = answer_question_with_gpt(question)  # Use GPT for answering
    print(f"Chatbot: {answer}")

print("Chatbot session ended.")