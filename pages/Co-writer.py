""" This file contains the code for the co-writer page.  This page allows
the user to chat with Luke Combs and receive guidance on their song writing.
This can be in the form of text or audio."""
import logging
# import asyncio
# from IPython.display import Audio
import streamlit as st
from dotenv import load_dotenv
# from utils.musicgen_utils import infer_endpoint
from utils.musicgen_utils import musicgen_pipeline
from dependencies import get_openai_client

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

client = get_openai_client()

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "cowriter_messages", "chat_state",
        "llm_inputs", "original_clip", "current_clip"
    ]
    default_values = [
        [], "text", None, None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state variables
init_cowriter_session_variables()

def chat_main():
    """ Main function for the chat page """
    if st.session_state.chat_state == "text":
        new_prompt = [{"role": "system", "content" : f"""
        You are Dave Matthews, the famous musician and songwriter, engaged in a
        co-writing session with the user who could be a fellow musician or fan.  The goal
        of the session is to help the user feel as though you are right alongside them,
        helping them craft their song with Dave's style and personality.  Do not break character.
        Your conversation so far is {st.session_state.cowriter_messages}.
        """}]
    elif st.session_state.chat_state == "audio":
        new_prompt = [{"role": "system", "content" : f"""
        You are Dave Matthews, the famous musician and songwriter, engaged in a
        co-writing session with the user who could be a fellow musician or fan.  The goal
        of the session is to help the user feel as though you are right alongside them,
        helping them craft their song with Dave's style and personality.  Do not break character.
        Your conversation so far is {st.session_state.cowriter_messages}  The user wants to also
        generate a music clip with the short description of {st.session_state.llm_inputs}.  There
        is a separate model that will handle the music generation, but go ahead and respond with
        a text related to the conversation and the new clip description to keep the conversation
        going while they wait for the new clip to be generated.  Let the user know that you are
        going to work on it and leave them something to think about while they wait.
        """}]

    # Set up the sidebar
    st.session_state.chat_state = st.sidebar.radio("Chat Mode", ["text", "audio"])
    st.text("")
    if st.session_state.current_clip:
        st.sidebar.markdown("**Current Audio Clip:**")
        # Display the current clip using the Audio component
        st.sidebar.audio(
            sample_rate=st.session_state.current_clip[1],
            data=st.session_state.current_clip[0]
        )
    if st.session_state.llm_inputs:
        st.sidebar.markdown("**Current Audio Gen Prompt:**")
        st.sidebar.markdown(st.session_state.llm_inputs)

    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Welcome to Co-writer!</h4>
    <br>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">While still in its early stages, we
        believe that Co-writer can become the gold standard for AI <i>enhancing</i>, rather than
        replacing, the artist's ability to engage with fans, fellow musicians, and others.  The goal
        is to make it feel as though the artist is really right there in the room, and we are getting
        closer every day to making that a reality.
        <br>
        <br>
        Without giving away too much, we have create a more simplified way to test out the very tip
        of the iceberg of what can be achieved with this technology.  You can engage with the artist
        in a text co-writing session back and forth, and in the very near
        future you will also be able to engage with the
        artist musically, generating audio clips based on
        both the context of the chat as well as uploaded clips.
        While this is far from the experience we aim to ultimately
        create, we hope that it will give you a small taste of what is to come.
        </h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Display markdown with animation
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                    -o-animation: fadeIn ease 3s; -ms-animation:
                    fadeIn ease 3s;">
                    </div>""", unsafe_allow_html=True)

    # Check if there are any messages in the session state
    if len(st.session_state.cowriter_messages) == 0:
        logging.debug("No messages in session state.")
        st.warning(
            "The audio generation does take some time, especially upon start.  As we scale,\
            we will continue to increase our compute thus speeding up the process dramatically.  However, for\
            demo purposes, we are not utilizing large amounts of GPU resources."
        )

    # Add a blank line
    st.text("")

    # Display chat messages from history on app rerun
    for message in st.session_state.cowriter_messages:
        logging.debug(f"Displaying message: {message}")
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🎸"):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Hey friend, let's start writing!"):
        logging.debug(f"Received user input: {prompt}")
        # Add user message to chat history
        st.session_state.cowriter_messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Load the prophet image for the avatar
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            message_placeholder = st.empty()
            full_response = ""
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages= new_prompt + [{"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.cowriter_messages],
            stream=True,
            temperature=0.75,
            max_tokens=350,
        )
        for chunk in response:
            if chunk.choices[0].finish_reason == "stop":
                logging.debug("Received 'stop' signal from response.")
                break
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.cowriter_messages.append({"role": "assistant", "content": full_response})
        if st.session_state.chat_state == "audio":
            with st.spinner("Composing your audio...  I'll be back shortly!"):
                st.session_state.current_clip = musicgen_pipeline()
                logging.info(f"Current clip: {st.session_state.current_clip}")
                logging.debug("Rerunning app after composing audio.")
                st.rerun()
    if st.session_state.current_clip:
        st.audio(sample_rate=st.session_state.current_clip[1], data=st.session_state.current_clip[0])

if __name__ == "__main__":
    logging.info("Starting main chat function.")
    chat_main()
