# Local RAG Application with Ollama, Langchain, and Milvus

This repository contains code for a running a local Retrieval Augmented Generation (RAG) application. It uses Ollama for LLM operations, Langchain for orchestration, and Milvus for vector storage.
I am using Llama 3 as the LLM at the moment. 

## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.11 or later
- Docker
- Docker-Compose

Additionally, you will need:
- An API key from Jina AI, which you can obtain [here](https://jina.ai).

## Installation

1. Clone this repository to your local machine:
```bash
git clone git@github.com:stephen37/ollama_local_rag.git
cd ollama_local_rag
```
2. Set Up Environment Variables:
Create a .env file in the root directory and add your Jina AI API key:
```
JINA_AI_API_KEY=your_jina_ai_api_key_here
```
3. Install dependencies 
```bash
poetry install
```
4. Start Milvus 
```bash
docker-compose up -d
```

## Usage
To run the application, execute the following command in your terminal:

```bash
python rag_ollama.py
```
You will be prompted to enter queries, and the system will retrieve relevant answers based on the data processed.

--- 
Feel free to check out [Milvus](https://github.com/milvus-io/milvus), and share your experiences with the community by joining our [Discord](https://discord.gg/FG6hMJStWu).