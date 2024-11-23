# AI-Powered Question Answering System

## Overview
This project implements an **AI-driven question-answering system** using a combination of **Hugging Face's Retrieval-Augmented Generation (RAG) model** and **OpenAI's GPT-3.5-turbo**. The system retrieves contextual answers from a custom dataset and integrates **FAISS** for efficient vector-based similarity search.

## Features
- **Custom Dataset**: Processes a custom dataset to create BERT embeddings for passage retrieval.
- **RAG Integration**: Combines retrieval with generative models to answer user questions.
- **OpenAI GPT Integration**: Provides fallback answers using OpenAI's GPT-3.5-turbo.
- **FAISS Indexing**: Ensures fast and accurate nearest-neighbor searches for embedding-based retrieval.
- **Dual-Answer Mechanism**: Combines RAG-based responses with GPT for robustness.

## Technologies Used
- **Programming Language**: Python  
- **Libraries and Tools**:
  - [Hugging Face Transformers](https://huggingface.co/transformers)
  - [FAISS (Facebook AI Similarity Search)](https://faiss.ai/)
  - [OpenAI API](https://platform.openai.com/)
  - PyTorch
  - Pandas
  - Datasets Library

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AyushS-03/AI-Powered-Question-Answering-System.git
   cd AI-Powered-Question-Answering-System
2. Set up OpenAI API:
    - Obtain your API key from **OpenAI**.

    - Add your API key to the **openai.api_key** variable in the script.

## Usage
1. Prepare your custom dataset:
    - Modify the data dictionary in the script with your own text dataset.

2. Run the script:
    ```bash
    python main.py

3. Interact with the chatbot:
    - Enter your question to receive a response.
    - Type exit or quit to end the session.