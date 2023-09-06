""" This file contains the code for the co-writer page.  This page allows
the user to chat with Luke Combs and receive guidance on their song writing.
This can be in the form of text or audio."""
import os
import numpy as np
import openai
import pinecone
import streamlit as st
from utils.model_utils2 import get_context, get_lyrics_vectorstore, get_inputs_from_llm, get_audio_sample, get_similar_audio_clips
from utils.encoder_utils import chunk_and_encode_encodec


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.success("**Welcome to co-writer!  There are two options for the chat:**\
                The first option is a standard text chat back and forth where\
                Luke can help you brainstorm ideas for your song.  The second\
                allows you to request that Luke help you compose an audio clip\
                based on your chat history and ideas generated.  You may switch\
                back and forth at any time to test them out!  Check out the sidebar\
                to make your selection.")
    st.markdown("---")
    with st.chat_message("assistant", avatar="🎸"):
        st.markdown("Hello, friend.  I'm excited to get our co-writing\
                    session started!  Why don't you tell me a little bit\
                    about the song you are working on?")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🎸"):
            st.markdown(message["content"])

response_type = st.sidebar.radio("Choose your response type",
                                ("Standard Chat", "Musical Chat"))

def get_text_response():
    """ Get a response from Luke Combs in the form of text."""
    if prompt := st.chat_input("Your message for Luke:"):
        with st.spinner("Luke is writing..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar = "🎸"):
                pinecone.init(api_key = os.getenv("PINECONE_KEY"),
                environment=os.getenv("PINECONE_ENV")) # Initialize pinecone
                vectorstore = get_lyrics_vectorstore(index_name = 'combs-data')
                context = get_context(vectorstore, prompt)
                context_dict = [{"Song Name" : context.page_content,
                                "lyrics" : context.metadata} for context in context]
                context = context_dict
                st.session_state.context = context
                messages = [
                    {
                        "role": "system", "content": f"""You are Luke Combs, the famous
                        country music singer, helping a fan out in a "co-writing" session
                        where you are giving them advice based on your own style to help 
                        them write songs.  You have context {context} pulled from your song
                        lyrics to help you relate to the user's question {prompt}.  Feel free
                        to mention a specific song or lyrics of yours when guiding the users along.
                        Your chat history so far is {st.session_state.chat_history}.  This will
                        be a back and forth chat, so make sure to leave your responses
                        open-ended."""
                    },
                    {
                        "role": "user", "content": f"""Please answer my {prompt} about 
                        song writing."""
                    },
                ]
                message_placeholder = st.empty()
                full_response = ""
                # Set list of models to iterate through
                models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k",
                        "gpt-3.5-turbo-0613", "gpt-3.5-turbo"] 
                for model in models:
                    try:
                        for response in openai.ChatCompletion.create(
                            model=model,
                            messages = messages,
                            max_tokens=250,
                            frequency_penalty=0.5,
                            presence_penalty=0.5,
                            temperature=1,
                            n=1,
                            stream=True
                        ):
                            full_response += response.choices[0].delta.get("content", "")
                            message_placeholder.markdown(full_response + "▌")
                            if response.choices[0].delta.get("stop"):
                                break
                        break 
                    except Exception as e:
                        print(e)
                        continue
                
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
  
def get_music_response():
    """ Get a response from Luke Combs in the form of an audio clip. """
  
    # Audio File Upload
    uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])
  
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
  
        # Audio Vector Encoding
        encoded_chunks = chunk_and_encode_encodec(audio_bytes)  # Replace with your actual encoding function
        audio_vector = np.mean(encoded_chunks, axis=0)

        
        # User Choice: Find Similar Songs or Generate Music
        user_choice = st.selectbox("What would you like to do next?", ["Find Similar Songs", "Generate Music"])
        
        if user_choice == "Find Similar Songs":
            # Similarity Search
            similar_clips = get_similar_audio_clips(audio_vector)  # Replace with your actual query function
            if similar_clips:
                st.write("Found similar clips from the artist:")
                # Display similar clips - this part depends on how your data is structured
            
        elif user_choice == "Generate Music":
            # Music Generation
            generated_music = generate_music(audio_vector)  # Replace with your actual music generation function
            st.audio(generated_music, format='audio/wav')

    # ... (Rest of your existing code)

    """ Get a response from Luke Combs in the form of an audio clip.
    We generate the clip by calling ChatGPT to create inputs based on 
    the chat history that are then fed into the MusicGen model to create
    an audio clip.  The audio clip is then returned to the user."""
    if prompt := st.chat_input("Your message for Luke:"):
        with st.spinner("Luke is composing... This will take a minute"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar = "🎸"):
                message_placeholder = st.empty()
                full_response = ""
                inputs = get_inputs_from_llm()
                output = get_audio_sample(inputs)
                message_placeholder.audio(np.array(output), sample_rate=22050)
            st.session_state.chat_history.append({"role": "assistant",
                                                "content": full_response})
                                                

# Create a button to reset the chat history
reset_button = st.sidebar.button("Reset Chat History", type="primary", use_container_width=True)
if reset_button:
    st.session_state.chat_history = []
    st.experimental_rerun()

if response_type == "Standard Chat":
    get_text_response()
else:
    get_music_response()
           