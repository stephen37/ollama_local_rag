import os
from pymilvus import MilvusClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain import hub

path_pdfs = "data/pdfs/"
documents = []
for file in os.listdir(path_pdfs):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(path_pdfs, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())


client = MilvusClient()
if client.has_collection("LangChainCollection"):
    client.drop_collection("LangChainCollection")


embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-de")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)

vectorstore = Milvus.from_documents(
    documents=all_splits,
    embedding=embeddings,
)


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
