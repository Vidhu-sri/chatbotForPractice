
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import openai
import logging
from htmlTemplates import css, bot_template, user_template





# Access the environment variable
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text
    

def get_text_chunks(raw_text):
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = CharacterTextSplitter(
        separator=".", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    for i, chunk in enumerate(chunks):
        if len(chunk) > chunk_size:
            print(f"Chunk {i} length: {len(chunk)}\n{chunk[:150]}...")
    return chunks


def get_vector_store(text_chunks):
    logging.info("Initializing HuggingFace Instruct Embeddings")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    logging.info("Creating FAISS vector store from texts")
    vectorstore = FAISS.from_texts(text_chunks, embedding = embeddings)
    logging.info("Vector store creation completed")
    return vectorstore
    

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory

    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with mutliple PDFS", page_icon=":books:")

    st.write(css, unsafe_allow_html= True)

    st.header("Chat With Multiple PDF's :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat history" not in st.session_state:
        st.session_state.chat_history = None

    
    


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'Process", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get chunks of text
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                #create vector store
                vectorstore = get_vector_store(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
  

if __name__ == '__main__':
    main()