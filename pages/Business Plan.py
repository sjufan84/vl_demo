""" Page for users to ask questions about the business plan
utilizing an LLM with a vectorstore as a retriever """
import uuid
import streamlit as st
from streamlit_chat import message
from utils.bplan_utils import get_bplan_response

if "bplan_chat_page" not in st.session_state:
    st.session_state.bplan_chat_page = "bplan_home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def bplan_intro():
    """ Page to introduce the business plan chat """
    st.session_state.chat_history = [] # Clear the chat history
    st.markdown("""### Business Plan Chat""")
    st.markdown("##### At Vocalockr, we're pioneering an interactive approach\
                to our business plan by using a vectorstore retriever and an LLM\
                for contextual answers. Though this technology is new, it promises\
                a more engaging experience. We're eager to discuss more in person!"
                )
    st.text("")
    start_bplan_chat_button = st.button("Start Business Plan Chat", type="primary",
                                         use_container_width=True)
    if start_bplan_chat_button:
        st.session_state.bplan_chat_page = "bplan_chat_main"
        st.experimental_rerun()

def bplan_chat_main():
    """ Main interface for the user to ask questions about the business plan """
    # Initial greeting from the ai
    initial_greeting = "Welcome to Vocalockr Business Plan Chat!  I'm their virtual\
        advisor and start-up strategist.  What questions do you have about their\
        plan?"
    if len(st.session_state.chat_history) == 0:
        message(initial_greeting, avatar_style="miniavs", seed="Socks")
    # Get the user's next question
    user_question = st.text_area("Your question", height=100)
    # Create a button to send the question to the vectorstore
    send_question_button = st.button("Send your question", type="primary", use_container_width=True)
    if send_question_button:
        with st.spinner("Getting response..."):
            # Get the bplan response
            get_bplan_response(user_question)
            # Set the session state to display the chat history
            st.session_state.bplan_chat_page = "bplan_chat_main"
            user_question = ""
            st.experimental_rerun()
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        # Display the chat history
        for chat_message in st.session_state.chat_history[-2:]:
            # If the role is "ai", display the message on the left
            if chat_message['role'] == "assistant":
                message(chat_message['content'], avatar_style="miniavs", seed="Socks",
                        key = f'{uuid.uuid4()}')
            # If the role is "user", display the message on the right
            elif chat_message['role'] == "user":
                message(chat_message['content'], avatar_style="miniavs", seed="Felix",
                        is_user=True, key=f'{uuid.uuid4()}')
                
    new_chat_button = st.button("Start New Chat Session", type="primary", use_container_width=True)
    if new_chat_button:
        st.session_state.chat_history = []
        st.experimental_rerun()
                

if st.session_state.bplan_chat_page == "bplan_home":
    bplan_intro()
elif st.session_state.bplan_chat_page == "bplan_chat_main":
    bplan_chat_main()