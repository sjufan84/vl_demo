""" This file contains the code for the co-writer page.  This page allows
the user to chat with Luke Combs and receive guidance on their song writing.
This can be in the form of text or audio."""
import asyncio
import streamlit as st
from dotenv import load_dotenv
from utils.model_utils import get_inputs_from_llm
from utils.musicgen_utils import get_music
from dependencies import get_openai_client

# Load environment variables
load_dotenv()

client = get_openai_client()

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "cowriter_messages", "openai_model", "chat_state",
        "llm_inputs", "original_clip", "current_clip"
    ]
    default_values = [
        [], "gpt-4-0613", "text", None, None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state variables
init_cowriter_session_variables()

async def chat_main():
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
    # uploaded_audio = None
    st.session_state.chat_state = st.sidebar.radio("Chat Mode", ["text"])
    st.text("")
    if st.session_state.chat_state == "audio":
        if isinstance(st.session_state.current_clip, str):
            st.sidebar.markdown("**Current Clip:**")
            st.sidebar.audio(st.session_state.current_clip, format="audio/wav", sample_rate=32000)
        # Create a file uploader for the user to be able to upload their own audio
    #    uploaded_audio = st.sidebar.file_uploader("Upload an audio clip", type=["wav", "mp3"])
    # if uploaded_audio and st.session_state.original_clip is None:
    #    original_clip, sr = librosa.load(uploaded_audio, sr=32000)
    #    st.session_state.current_clip = st.session_state.original_clip
    # if st.session_state.current_clip is not None:
    #    st.sidebar.markdown("**Current Clip:**")
    #    st.write(st.session_state.current_clip)
    # st.sidebar.audio(st.session_state.current_clip, format="audio/wav", sample_rate=32000)

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
        Without giving away too much, we have created a much simplified way to test out the very tip
        of the iceberg of what can be achieved with this technology.  You can engage with the artist
        in a text co-writing session back and forth, and in the very near future you will also be able to engage with the 
        artist musically, generating audio clips based on both the context of the chat as well as uploaded clips.
        While this is far from the experience we aim to ultimately create, we hope that it will give you a small taste of what is to come.
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
    # if len(st.session_state.cowriter_messages) == 0:
    #    st.warning("The audio generation does take some time, especially upon start.  As we scale,\
    #    we will continue to increase our compute thus speeding up the process dramatically.  However, for\
    #    demo purposes, we are not utilizing large amounts of GPU resources.")
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
        st.session_state.llm_inputs = get_inputs_from_llm()
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
            break
          full_response += chunk.choices[0].delta.content
          message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.cowriter_messages.append({"role": "assistant", "content": full_response})
        if st.session_state.chat_state == "audio":
            with st.spinner("Composing your audio...  I'll be back shortly!"):
                st.session_state.current_clip = await get_music(st.session_state.llm_inputs)
                st.experimental_rerun()

if __name__ == "__main__":
    asyncio.run(chat_main())
