from langchain_community.document_loaders import UnstructuredXMLLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.milvus import Milvus
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain import hub
from langchain_experimental.text_splitter import SemanticChunker

import os

path_xmls = "data/french_parliament/compteRendu/"
documents = []
for file in os.listdir(path_xmls):
    if file.endswith(".xml"):
        pdf_path = os.path.join(path_xmls, file)
        loader = UnstructuredXMLLoader(pdf_path)
        documents.extend(loader.load())


embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

text_splitter = SemanticChunker(embeddings)
all_splits = text_splitter.split_documents(documents)

vectorstore = Milvus.from_documents(documents=all_splits, embedding=embeddings, collection_name="french_parliament")

def run_query() -> None:
    llm = Ollama(
        model="mistral",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
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
