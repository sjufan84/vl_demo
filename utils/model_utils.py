""" Loading lyrics and providing model context for the chatbot """
import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
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
pinecone.init(api_key = os.getenv("PINECONE_KEY2"), environment=os.getenv("PINECONE_ENV2")) # Initialize pinecone

from langchain.embeddings.openai import OpenAIEmbeddings

embed = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key = openai.api_key,
    disallowed_special=()
)



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
    vectorstore = create_pinecone_vectorstore(load_lyrics()) # Create the vectorstore
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

def get_bplan_response(question: str):
    """Get a response from the business plan to the question"""
    vectorstore = get_vectorstore()
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