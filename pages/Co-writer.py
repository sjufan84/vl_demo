""" UI for the user to be able to interact with the artist Luke Combs """
import uuid
import streamlit as st
from streamlit_chat import message
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
from utils.chat_utils import add_message
from utils.model_utils import get_luke_response 

# Define the page config
st.set_page_config(page_title="Luke Combs Chat", initial_sidebar_state="collapsed")

if "chat_page" not in st.session_state:
    st.session_state.chat_page = "chat_home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def chat_home():
    """ Home page for chat with training pipeline image """
    # Insert the training pipeline image
    training_pipeline_image = Image.open("images/training_pipeline.png")
    st.image(training_pipeline_image, use_column_width=True)

    st.text("")
    st.markdown("**Above we can see the basic training pipeline for creating\
                boutique LLMs that are tailored to the specific artist and can\
                be used downtstream for a variety of applications.**")
    
    st.markdown("**For a simple demo, we have created a chatbot that takes\
                on the persona of the artist Luke Combs.  You can engage\
                with Luke to get his thoughts on your songwriting.  Obviously,\
                the uniquely trained model would be much more robust, but\
                this demo should give you a sense of the potential.**")
    # Create the button to start the chat
    start_chat_button = st.button("Start Co-writing with Luke", type="primary",
                                  use_container_width=True)
    if start_chat_button:
        # Switch to the chat intro page
        st.session_state.chat_page = "chat_intro"
        st.experimental_rerun()

def chat_intro():
    """ The home page for the artist chat interactions """
    st.markdown("##### Welcome to Co-writer!  Luke Combs\
                is here to help guide you and encourage you\
                as you write your next hit song!")
    st.markdown("---")
    # Display an initial greeting from the artist
    initial_greeting = "Hey there!  I'm Luke Combs, and I'm\
                        excited to co-write with you!  Why don't you tell me\
                        about the song you're working on?"
    message(initial_greeting, avatar_style="initials", seed="LC")
   
    # Add the initial greeting to the chat history
    add_message("ai", initial_greeting)
    # Get the user's question
    user_question = st.text_area("Your message for Luke", height=100)
    # Create a button to send the question to the artist
    send_question_button = st.button("Send Your Message to Luke", type="primary",
                                     use_container_width=True)
    if send_question_button:
        with st.spinner("Luke is writing..."):
            # Get the artist's response
            get_luke_response(user_question)
            # Set the session state to display the chat history
            st.session_state.chat_page = "display_chat"
            st.experimental_rerun()    
    

def display_chat():
    """ Display the ongoing chat """
    # Get the user's next question
    user_question = st.text_area("Your message for Luke", height=100)
    # Create a button to send the question to the artist
    send_question_button = st.button("Send Message to Luke", type="primary", use_container_width=True)
    if send_question_button:
        with st.spinner("Luke is typing..."):
            # Get the artist's response
            get_luke_response(user_question)
            # Set the session state to display the chat history
            st.session_state.chat_page = "display_chat"
            user_question = ""
            st.experimental_rerun()
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        # Display the chat history
        for chat_message in st.session_state.chat_history[-2:]:
            # If the role is "ai", display the message on the left
            if chat_message['role'] == "assistant":
                message(chat_message['content']['result'], avatar_style="initials", seed="LC",
                        key = f'{uuid.uuid4()}')
            # If the role is "user", display the message on the right
            elif chat_message['role'] == "user":
                message(chat_message['content'], avatar_style="initials", seed="You",
                        is_user=True, key=f'{uuid.uuid4()}')

    st.markdown("---")
    # Create a button to start a new chat
    new_chat_button = st.button("New Chat", type="primary", use_container_width=True)
    if new_chat_button:
        # Clear the chat history
        st.session_state.chat_history = []
        # Switch to the chat home page
        st.session_state.chat_page = "chat_home"
        st.experimental_rerun()
    
    # Create a button to go back to the demo home page
    home_button = st.button("Home", type="primary", use_container_width=True)
    if home_button:
        # Clear the chat history
        st.session_state.chat_history = []
        # Switch to the chat home page
        st.session_state.chat_page = "chat_home"
        switch_page("Demo Visualize")

# Set the flow of the page
if st.session_state.chat_page == "chat_home":
    chat_home()
elif st.session_state.chat_page == "chat_intro":
    chat_intro()
elif st.session_state.chat_page == "display_chat":
    display_chat()
