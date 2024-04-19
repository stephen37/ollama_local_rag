from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.milvus import Milvus
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain import hub

import os
from dotenv import load_dotenv


load_dotenv()
JINA_AI_API_KEY = os.getenv("JINA_AI_API_KEY")

loader = PyPDFLoader(
    "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf"
)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


embeddings = JinaEmbeddings(
    jina_api_key=JINA_AI_API_KEY, model_name="jina-embeddings-v2-base-en"
)
vectorstore = Milvus.from_documents(documents=all_splits, embedding=embeddings)


def run_query() -> None:
    llm = Ollama(
        model="llama3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"],
    )

    query = input("\nQuery: ")
    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain({"query": query})
    print(result)


if __name__ == "__main__":
    while True:
        run_query()
