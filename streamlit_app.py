import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ========================
# 1️⃣ Configuration
# ========================
# Load environment variables and API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# ========================
# 2️⃣ File Size Limits
# ========================
MAX_TOTAL_SIZE_MB = 5
MAX_FILE_SIZE_MB = 2

def validate_file_sizes(uploaded_files):
    total_size = 0
    for file in uploaded_files:
        size_mb = file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.warning(f"{file.name} is too large ({size_mb:.2f} MB). Limit is {MAX_FILE_SIZE_MB} MB per file.")
            return False
        total_size += size_mb

    if total_size > MAX_TOTAL_SIZE_MB:
        st.warning(f"Total size of uploaded files is {total_size:.2f} MB. Limit is {MAX_TOTAL_SIZE_MB} MB in total.")
        return False

    return True

# ========================
# 3️⃣ Text Extraction Functions
# ========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def get_html_text(html_file):
    content = html_file.read()
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()

# ========================
# 4️⃣ Text Chunking and Vector Store
# ========================
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ========================
# 5️⃣ Conversational Chain Setup
# ========================
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, say "answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# ========================
# 6️⃣ Streamlit App Layout
# ========================
def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("Chat with your PDF, DOCX, or HTML using Gemini 💬")

    user_question = st.text_input("Ask a question about your uploaded files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload & Process Files")
        uploaded_files = st.file_uploader("Upload PDF, DOCX, or HTML files", accept_multiple_files=True, type=['pdf', 'docx', 'html'])

    


        if st.button("Submit & Process"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
                return

            if not validate_file_sizes(uploaded_files):
                return

            with st.spinner("Processing files..."):
                full_text = ""
                for file in uploaded_files:
                    if file.name.endswith(".pdf"):
                        full_text += get_pdf_text([file])
                    elif file.name.endswith(".docx"):
                        full_text += get_docx_text(file)
                    elif file.name.endswith(".html"):
                        full_text += get_html_text(file)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")

                text_chunks = get_text_chunks(full_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

if __name__ == "__main__":
    main()
