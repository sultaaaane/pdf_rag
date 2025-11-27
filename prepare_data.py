from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

BOOKS_PATH = "./books"


def load_documents():
    loader = DirectoryLoader(BOOKS_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


docs = load_documents()
if not docs:
    print(f"No documents found in {BOOKS_PATH}")
else:
    all_splits = text_splitter.split_documents(docs)
    print(all_splits[10])
