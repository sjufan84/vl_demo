""" Loading lyrics and providing model context for the chatbot """
import os
import pandas as pd
import pinecone
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import openai
import streamlit as st
from dotenv import load_dotenv
from utils.chat_utils import add_message, ChatMessage

load_dotenv() # Load the .env file

# Read in the openai api key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
# Read in the openai organization id from the .env file
openai.organization = os.getenv("OPENAI_ORG")

embed = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key = openai.api_key,
    disallowed_special=()
)

if "context" not in st.session_state:
    st.session_state.context = []

def get_vectorstore(index_name='vocalockr-bplan', embeddings = embed):
    """Get the vectorstore from Pinecone"""
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    
    return vectorstore

def custom_csv_loader(file_path):
    """ Load the csv file and return a dataframe """
    return pd.read_csv(file_path, encoding='utf-8') 

@st.cache_data
def load_lyrics():
    """Load the lyrics from the csv file"""
    df = pd.read_csv('./lyrics/combs_lyrics.csv', encoding='utf-8')
    loader = DataFrameLoader(df, page_content_column="Song Title") # noqa

    data = loader.load() # Load the data

    return data

@st.cache_data


def get_luke_response(question:str):
    """Get a response from Luke Combs to the question"""
    pinecone.init(api_key = os.getenv("PINECONE_KEY"),
    environment=os.getenv("PINECONE_ENV")) # Initialize pinecone
    vectorstore = get_vectorstore(index_name = 'combs-data')
    context = get_context(vectorstore, question)
    context_dict = [{"Song Name" : context.page_content, "lyrics" : context.metadata} for context in context]
    context = context_dict
    st.session_state.context = context
    messages = [
        {
            "role": "system", "content": f"""You are Luke Combs, the famous
            country music singer, helping a fan out in a "co-writing" session
            where you are giving them advice based on your own style to help 
            them write songs.  You have context {context} pulled from your song
            lyrics to help you relate to the user's question {question}.  Feel free
            to mention a specific song or lyrics of yours when guiding the users along.
            Your chat history so far is {st.session_state.chat_history}.  This will
            be a back and forth chat, so make sure to leave your responses open-ended."""
        },
        {
            "role": "user", "content": f"""Please answer my {question} about 
            song writing."""
        },
    ]
    models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613, gpt-3.5-turbo"] # Set list of models to iterate through
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages = messages,
                max_tokens=500,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1
            )
            answer = response.choices[0].message.content
            add_message("user", question)
            # Format the user question and the AI answer into ChatMessage objects
            ai_answer = ChatMessage(answer, "ai")
            # Add the user question and AI answer to the chat history
            add_message(ai_answer.role, ai_answer.content)

            return answer
        
        except Exception as e:
            print(e)
            continue

def get_bplan_response(question: str):
    """Get a response from the business plan to the question"""
    pinecone.init(api_key = os.getenv("PINECONE_KEY2"), environment=os.getenv("PINECONE_ENV2")) # Initialize pinecone
    vectorstore = get_vectorstore(index_name='vocalockr-bplan')
    context = get_context(vectorstore, question)
    messages = [
        {
            "role": "system", "content": f"""You are a master busines advisor
            and start-up strategist answering a question {question} about 
            an early stage company's business plan.  The relevant information
            from the business plan is {context}.  Your chat history so far is
            {st.session_state.chat_history}."""
        },
        {
            "role": "user", "content": f"""Please answer my {question} about the 
            business plan."""
        },
    ]
    models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613, gpt-3.5-turbo"] # Set list of models to iterate through
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages = messages,
                max_tokens=500,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1
            )
            answer = response.choices[0].message.content
            add_message("user", question)
            # Format the user question and the AI answer into ChatMessage objects
            ai_answer = ChatMessage(answer, "ai")
            # Add the user question and AI answer to the chat history
            add_message(ai_answer.role, ai_answer.content)

            return answer
        
        except Exception as e:
            print(e)
            continue

def get_context(vectorstore, question: str):
    """Get the context from the vectorstore"""
    context = vectorstore.similarity_search(
    query=question,
    k=3
    )

    return context