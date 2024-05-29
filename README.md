# Local RAG Application with Ollama, Langchain, and Milvus

This repository contains code for running local Retrieval Augmented Generation (RAG) applications. It uses Ollama for LLM operations, Langchain for orchestration, and [Milvus](https://github.com/milvus-io/milvus) for vector storage, it is using Llama3 for the LLM.

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
2. Install dependencies 
```bash
poetry install
```
3. Start Milvus with Docker
```bash
docker-compose up -d
```

## Usage
To run the different applications, execute the following command in your terminal:

```bash
python <file_name.py>
```
You will be prompted to enter queries, and the system will retrieve relevant answers based on the data processed.

For example, if you want to interact with the data from the French parliament, you can run `python rag_french_parliament.py` 

--- 
Feel free to check out [Milvus](https://github.com/milvus-io/milvus), and share your experiences with the community by joining our [Discord](https://discord.gg/FG6hMJStWu).