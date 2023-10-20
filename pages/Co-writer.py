""" This file contains the code for the co-writer page.  This page allows
the user to chat with Luke Combs and receive guidance on their song writing.
This can be in the form of text or audio."""
import os
from io import BytesIO
import asyncio
import openai
import streamlit as st
import torchaudio
from dotenv import load_dotenv
from utils.model_utils import get_inputs_from_llm
from utils.musicgen import get_music

# Load environment variables
load_dotenv()

# Set OpenAI API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_KEY2")
openai.organization = os.getenv("OPENAI_ORG2")

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = ["cowriter_messages", "openai_model", "chat_state", "original_clip", "current_audio_clip"
    ]
    default_values = [None, "gpt-4-0613", "text", None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state variables
init_cowriter_session_variables() 

def chat_main():
    """ Main function for the chat page """
    new_prompt = [{"role": "system", "content" : f"""
    You are Dave Matthews, the famous musician and songwriter, engaged in a
    co-writing session with the user who could be a fellow musician or fan.  The goal
    of the session is to help the user feel as though you are right alongside them,
    helping them craft their song with Dave's style and personality.  Do not break character.
    Your conversation so far is {st.session_state.cowriter_messages}. 
    """}]

    # Set up the sidebar
    st.session_state.chat_state = st.sidebar.radio("Chat Mode", ["text", "audio"])
    st.text("")
    if st.session_state.chat_state == "audio":
        # Create a file uploader for the user to be able to upload their own audio
        uploaded_audio = st.sidebar.file_uploader("Upload an audio clip", type=["wav", "mp3"])
        if uploaded_audio:
            st.session_state.original_clip = torchaudio.load(BytesIO(uploaded_audio.getvalue()))
    if st.session_state.original_clip != None:
        st.sidebar.markdown("Original Audio Clip:")
        st.sidebar.audio(st.session_state.original_clip[0].numpy(), sample_rate=st.session_state.original_clip[1], format="audio/wav", start_time=0)
    if st.session_state.current_audio_clip == None and st.session_state.original_clip != None:
        st.session_state.current_audio_clip = st.session_state.original_clip
    if st.session_state.current_audio_clip != None and st.session_state.current_audio_clip != st.session_state.original_clip:
        st.markdown("Current Audio Clip:")
        st.sidebar.audio(st.session_state.current_audio_clip, format="audio/wav", start_time=0)
    if not st.session_state.cowriter_messages:
        st.session_state.cowriter_messages = new_prompt

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
        believe that Co-writer can become the gold standard for AI <i>enhancing</i> rather than
        replacing, the artist's ability to engage with fans, fellow musicians, and others, at scale,
        and in a way that is authentic and true to the artist's style and personality.
        <br>
        <br>
        Without giving away too much, we have created a simplified way to test out the very tip
        of the iceberg of what can be achieved with this technology.  You can chat with the artist below
        for tips on your song-writing journey, or, if you so chooose, you can upload your own
        audio clips and a completely original piece of music will be generated based on the clip and your
        conversation history.  Select an option on the sidebar to get started, and have fun!
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
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                -o-animation: fadeIn ease 3s; -ms-animation:
                fadeIn ease 3s;">
                </div>""", unsafe_allow_html=True)
    st.text("")
    # Display chat messages from history on app rerun
    for message in st.session_state.cowriter_messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🎸"):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Hey friend, let's start writing!"):
        # Add user message to chat historys
        st.session_state.cowriter_messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Load the prophet image for the avatar
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            message_placeholder = st.empty()
            full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages= [{"role": m["role"], "content": m["content"]} for m in st.session_state.cowriter_messages],
            stream=True,
            temperature=1,
            max_tokens=200,
            ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.cowriter_messages.append({"role": "assistant", "content": full_response})
        if st.session_state.chat_state == "audio":
            with st.spinner("Composing your audio..."):
                st.session_state.current_audio_clip = get_music(st.session_state.current_audio_clip, get_inputs_from_llm())

chat_main()