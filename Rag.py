from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains.question_answering import load_qa_chain

CHROMA_PATH = "./db/chroma_langchain_db"
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_text(text)


if __name__ == "__main__":
    pdf = st.file_uploader("Upload your files here")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        chunks = split_text(text)

        knowledge_base = FAISS.from_texts(chunks, EMBEDDING_FUNCTION)
        user_input = st.text_input("Write your question for your pdf")

        if user_input:
            docs = knowledge_base.similarity_search(user_input)

            llm = OllamaLLM(model="llama3.2")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": user_input})
            st.write(response["output_text"])
