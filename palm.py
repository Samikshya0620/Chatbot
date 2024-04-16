import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import sys

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def setup_chatbot(file_path):
    llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200})
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(load_pdf(file_path))
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory='./data')
    vectordb.persist()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={'k': 7}), return_source_documents=True)
    return ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(search_kwargs={'k': 6}), return_source_documents=True), qa_chain



def update_user_info(name, phone_number, email):
    user_info = {
        'name': name,
        'phone_number': phone_number,
        'email': email
    }
    with open("user_info.txt", "a") as f:
        for key, value in user_info.items():
            f.write(f"{key}: {value}\n")

    

counter = 0


def get_unique_key():
    global counter
    counter += 1
    return f"input_{counter}"

user_info = {'data': []}


def main():
    st.sidebar.title("Chatbot Info")
    st.sidebar.write("- Using LangChain\n - Using HuggingFace Embeddings \n - Model 'google/flan-t5-large' is a T5 (Text-to-Text Transfer Transformer) model trained by Google.")
    st.title("Chat with your Documents")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_file is not None:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")
        st.write("Initializing chatbot...")
        global qa_chain
        qa_chain, conversational_chain = setup_chatbot(file_path)

        name = st.text_input("Your Name")
        phone_number = st.text_input("Your Phone Number")
        email = st.text_input("Your Email")

        if st.button("Update User Info"):
            update_user_info(name, phone_number, email)

    
        st.write("Chatbot initialized. Start chatting!")
        
        chat_history=[]
        while True:
            query = st.text_input("Prompt:", key=get_unique_key())
            if query.lower() in ["exit", "quit", "q"]:
                st.write("Exiting")
                sys.exit()
            else:
                
                result = conversational_chain({'query': query, 'chat_history': chat_history})

                chat_history.append((query, result['result']))

                st.write('Answer: ' + result['result'])



if __name__ == "__main__":
    main()