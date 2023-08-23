""" Loading lyrics and providing model context for the chatbot """
import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import openai
import streamlit as st
from dotenv import load_dotenv
from utils.chat_utils import add_message, ChatMessage

load_dotenv() # Load the .env file

# Read in the openai api key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
# Read in the openai organization id from the .env file
openai.organization = os.getenv("OPENAI_ORG")

def custom_csv_loader(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

@st.cache_data
def load_lyrics():
    """Load the lyrics from the csv file"""
    df = pd.read_csv('./lyrics/combs_lyrics.csv', encoding='utf-8')
    loader = DataFrameLoader(df, page_content_column="Song Title") # noqa

    data = loader.load() # Load the data

    return data

@st.cache_resource
def get_vectorstore():
    """Get the vectorstore from the lyrics"""
    data = load_lyrics() # Load the lyrics

    vectorstore = Chroma.from_documents(data, OpenAIEmbeddings()) # Get the vectorstore

    return vectorstore

@st.cache_data
def create_prompt():
    """Create the prompt for the chatbot"""
    template = """You are Luke Combs, the famous country music singer, 
    helping a fan as their 'co-writer'. Use the following pieces of context
    containing lyrics to your songs to help guide the fan in writing their
    song, bringing your own unique style and voice to the sesssion.  Keep
    your answers concise and focused on helping the fan write their song.
    {context}
    Question: {question}
    Helpful Answer:"""
    qa_chain_prompt = PromptTemplate.from_template(template)

    return qa_chain_prompt

def get_luke_response(question:str):
    """Get a response from Luke Combs to the question"""
    vectorstore = get_vectorstore()
    qa_chain_prompt = create_prompt()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9,
                     max_tokens=300)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": qa_chain_prompt}, verbose=True)
    response = qa_chain({"query": question})

    add_message("user", question)
    # Format the user question and the AI answer into ChatMessage objects
    ai_answer = ChatMessage(response, "ai")
    # Add the user question and AI answer to the chat history
    add_message(ai_answer.role, ai_answer.content)

    return response