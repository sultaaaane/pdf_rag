import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BOOKS_PATH = "./books"
CHROMA_PATH = "./db/chroma_langchain_db"


def get_embeddings(embedding_function):

    embeddings = embedding_function
    return embeddings


def store_vectore_db(docs):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = get_embeddings(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    db = Chroma(
        collection_name="Books",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    db.add_documents(documents=docs)
    result = db.similarity_search("what is the context")
    print(result)


def load_documents():
    loader = DirectoryLoader(BOOKS_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


docs = load_documents()
if not docs:
    print(f"No documents found in {BOOKS_PATH}")
else:
    all_splits = split_text(docs)
    store_vectore_db(docs)
