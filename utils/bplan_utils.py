""" Utils related to the business plan features. """
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone   
import pinecone
import openai
import streamlit as st
from utils.chat_utils import add_message

load_dotenv()

# Load the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_org = os.getenv("OPENAI_ORG")

pinecone_key = os.getenv("PINECONE_KEY2")
pinecone_env = os.getenv("PINECONE_ENV2")

embeddings = OpenAIEmbeddings()

def get_vectorstore(index_name:str = "vlockr-bplan"):
    """ Get the vectorstore from Pinecone. """
    # Connect to the vector database
    pinecone.init(api_key=pinecone_key, environment=pinecone_env)
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    
    return vectorstore

def query_bplan(query: str):
    """ Perform a query on the business plan vector database. """
    vectorstore = get_vectorstore()
    # Connect to the vector database
    context = vectorstore.similarity_search(
    query=query,
    k=3
    )

    return context
    
def get_bplan_response(question:str):
    """Get a response from the bplan to the question"""
    # Create a list of messages to send to the LLM based on the doc.page_content
    context = query_bplan(question)
    context = [context.page_content for context in context]
    messages = [
        {
            "role": "system", "content": f'''You are a master start-up advisor\
            providing analysis and question answering about a business plan.\
            The relevant context for your response is the following: {context}.\
            Based on the provided context, do your best to answer  the user's\
            question {question} about the business plan.  Your most recent chat history is
            {st.session_state.chat_history[-3:]}.'''
        },
        {
            "role": "user", "content": f"Please answer my {question} about the business plan."
        }
    ]
    # Add the user's question to the chat history
    add_message("user", question)
    
    models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo"]
    
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=500,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1,
            )
            full_response = response.choices[0].message.content

            # Add the response to the chat history
            add_message("ai", full_response)
                
            return full_response
        
        except Exception as e:
            print(e)
            continue
