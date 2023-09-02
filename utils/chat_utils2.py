"""
The primary chat utilities
"""
import os
import requests
import streamlit as st
import openai
from dotenv import load_dotenv


# Load the environment variables
load_dotenv()

# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Initialize a connection to the redis st

# Define a class for the chat messages
class ChatMessage:
    # Define the init method
    def __init__(self, content, role):
        self.content = content
        self.role = role

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_message(role, content):
    """ Add a message to the chat history.
     Args:
        role (str): The role of the message.  Should be one of "user", "ai", or "system"
        content (str): The content of the message
    """
    # If the role is user, we will add a user message formatted as a HumanMessage
    if role == "user":
        message = {"role": "user", "content": content}
    # If the role is ai, we will add an ai message formatted as an AIMessage
    elif role == "ai":
        message = {"role": "assistant", "content": content}
    # If the role is system, we will add a system message formatted as a SystemMessage
    elif role == "system":
        message = {"role": "system", "content": content}
    # Append the message to the chat history
    st.session_state.chat_history.append(message)