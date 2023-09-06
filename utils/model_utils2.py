""" Loading lyrics and providing model context for the chatbot """
import os
#import logging
import pandas as pd
import numpy as np
import pinecone
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import openai
import streamlit as st
from typing import Union
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from utils.chat_utils import add_message


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
# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
       'context', 'output', 'prompt'
    ]
    default_values = [
        [], None, ""
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()


def get_lyrics_vectorstore(index_name='combs-data', embeddings = embed):
    """Get the vectorstore from Pinecone"""
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    
    return vectorstore

def get_music_vectorstore(index_name='combs-clips'):
    """Get the vectorstore from Pinecone"""
    api_key = os.getenv("PINECONE_KEY2")
    environment = os.getenv("PINECONE_ENV2")
    pinecone.init(api_key = api_key, environment=environment)
    music_vectorstore = pinecone.Index(index_name)

    return music_vectorstore
    
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
    environment=os.getenv("PINECONE_ENV"))  # Initialize pinecone
    vectorstore = get_lyrics_vectorstore(index_name='combs-data')
    context = get_context(vectorstore, question)
    context_dict = [{"Song Name": context.page_content, "lyrics": context.metadata} for context in context]
    st.session_state.context = context_dict  # Cache the context
    
    messages = [
        {
            "role": "system", "content": f"""You are Luke Combs, the famous
            country music singer, helping a fan out in a "co-writing" session
            where you are giving them advice based on your own style to help 
            them write songs. You have context {context_dict} pulled from your 
            song lyrics to help you relate to the user's question {question}. 
            Feel free to mention a specific song or lyrics of yours when guiding the users along.
            Your chat history so far is {st.session_state.chat_history}. 
            This will be a back and forth chat, so make sure to leave your responses open-ended."""
        },
        {
            "role": "user", "content": f"Please answer my {question} about song writing."
        }
    ]
    
    models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo"]
    full_response = ""
    
    for model in models:
        try:
            for response in openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=500,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1,
                stream=True
            ):
                full_response += response.choices[0].delta.get("content", "")
                if response.choices[0].delta.get("stop"):
                    break
            return full_response
        
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

def get_inputs_from_llm():
    """ We want the LLM to decide on the prompts for the 
    music generation model"""
    messages = [
        {
            "role": "system", "content": f"""You are Luke Combs, the famous
            country music singer, helping a fan out in a "co-writing" session
            where you are giving them advice based on your own style to help 
            them write songs.  The user would like you to help them create an
            audio sample based on your chat history {st.session_state.chat_history}
            so far.  Based on the chat history, create a short text prompt that
            would best identify to the music generation model what kind of song
            to create.  For example, if the chat history is about love and heartbreak,
            you could say "love, heartbreak, slow, country" to the music generation
            model.  Also include any specific instruments, beats, or other details
            that the user wants to specify in the prompt.  Assume that the style of the
            song will be based on your own style, unless otherwise noted.  The prompt
            should be formatted as a string of attributes separated by commas.  For example,
            "love, heartbreak, slow, country, guitar, piano, drums, bass, upbeat, sad, happy.
            Do not return anything other than the prompt."""
        },
        {
            "role": "user", "content": f"""Please help me create a prompt for the
            music generation model based on our chat history."""
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
            st.session_state.prompt = answer
            return answer
        
        except Exception as e:
            print(e)
            continue


def get_audio_sample(inputs: str, audio_file: Union[str, bytes] = None):
    """Get an audio sample from the music gen model
    based on the chat history"""
    if audio_file:
        print("Audio file provided")
    
    client = InferenceClient(model = "https://sz8hb6gcq2ersref.us-east-1.aws.endpoints.huggingface.cloud",
                        token=os.getenv("INFERENCE_KEY")) # Initialize the inference client
    # We want the LLM to decide on the prompts
    json = {"inputs": inputs}
    response = client.post(json=json)
    # response looks like this b'[{"generated_text":[[-0.182352,-0.17802449, ...]]}]
    output = eval(response)[0]["generated_audio"]
    add_message(role="ai", content="Here is the audio sample I created for you.\
                Let me know what you think!")
    st.session_state.output = output

    return output

def get_similar_audio_clips(audio_vector: Union[list, np.array]):
    """Get the most similar audio clips from the music vectorstore"""
    music_vectorstore = get_music_vectorstore(index_name='combs-clips')
    similar_audio = music_vectorstore.query(
        vector=audio_vector,
        k=3, 
        include_metadata=True
    )

    return similar_audio
