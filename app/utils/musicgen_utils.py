import logging
# import os
import numpy as np
import streamlit as st
from dependencies import get_openai_client
# from huggingface_hub import InferenceClient
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize session state for LLM inputs if not already present
if "llm_inputs" not in st.session_state:
    st.session_state.llm_inputs = None

# Define API URL and headers for Hugging Face model
API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
headers = {"Authorization": "Bearer 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"}

@st.cache_resource
def get_pipe():
    music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small")

    return music_generator

async def get_inputs_from_llm(artist: str = "Dave Matthews"):
    """
    Function to get inputs from LLM. It uses the OpenAI client
    to generate a prompt for the music generation model.
    """
    logging.info("Getting inputs from LLM.")
    client = get_openai_client()
    messages = [
        {
            "role": "system", "content": f"""You are {artist}, the famous
            musician, helping a fan out in a "co-writing" session
            where you are giving them advice based on your own style to help
            them write songs.  The user would like you to help them create an
            audio sample based on your chat history {st.session_state.cowriter_messages}
            so far.  Based on the chat history, create a text prompt that
            would best identify to the music generation model what kind of song
            to create.  Remember you are {artist} when contemplating your answer.Think through
            how you can best represent the song you are helping the user create in your prompt.
            The prompt should be no longer than the examples below:\n\
                1) "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms,
                perfect for the beach."
                2) "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and
                soaring strings, creating a cinematic atmosphere fit for a heroic battle."
                3) "reggaeton track, with a booming 808 kick, synth melodies layered with
                Latin percussion elements, uplifting and energizing"
                4) "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions"
                """
        },
    ]
    models = ["gpt-4-1106-preview", "gpt-4-0613", "gpt-3.5-turbo-1106"]
    for model in models:
        try:
            # Try to generate a completion with the current model
            response = client.chat.completions.create(
                model=model,
                messages = messages,
                max_tokens=50,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=0.75,
                n=1
            )
            answer = response.choices[0].message.content
            st.session_state.prompt = answer
            st.session_state.llm_inputs = answer
            return answer

        except TimeoutError as e:
            logging.error(f"Timeout error: {e}")
            continue

'''async def get_audio_clip(inputs: str = None, max_retries: int = 5, wait_time: int = 15):
    """
    Function to get audio clip from the music generation model.
    It sends a POST request to the Hugging Face API and handles potential errors.
    """
    wait_time = 120
    normal_time = 15
    logging.info("Getting audio clip.")
    if inputs is None:
        inputs = get_inputs_from_llm()

    # initially set the "wait_for_model" parameter to False
    # this will be set to True if we get a 503 error
    wait_for_model = False

    # payload = {"inputs": inputs, "parameters": {"wait_for_model": wait_for_model}}
    retries = 0

    while retries < max_retries:
        try:
            payload = {"inputs": inputs, "options" : {"wait_for_model": wait_for_model}}
            # Try to send a POST request to the API
            logging.info(f"Payload: {payload}")
            response = requests.request("POST", API_URL, headers=headers, json=payload)
            # If the response status code is 503, sleep for the "normal_time" seconds
            # And set the "wait_for_model" parameter to True
            if response.status_code == 503:
                logging.warning(f"Response status code: {response.status_code}")
                logging.info(f"Waiting for {wait_time} seconds.")
                wait_for_model = True
                time.sleep(wait_time)
                retries += 1
                continue
            # If the response status code is 200, return the response content
            elif response.status_code == 200:
                logging.info(f"Response status code: {response.status_code}")
                logging.debug(f"Response: {response}")
                logging.debug(f"Response content: {response.content}")
                return response.content
            # If the response status code is neither 200 nor 503, raise an error
            else:
                logging.error(f"Response status code: {response.status_code}")
                raise Exception("Error while calling API.")
        except Exception as e:
            logging.error(f"Exception: {e}")
            retries += 1
            continue

    # If we reach the max number of retries, raise an error
    logging.error("Max number of retries reached.")
    raise Exception("Max number of retries reached.")'''

async def musicgen_pipeline(prompt: str = None):
    """ Function to generate music clip based on user input """
    # Get the music generation pipeline
    music_generator = get_pipe()
    logging.info(f"Music generator: {music_generator}")

    # Get the inputs for the LLM
    if prompt is None:
        prompt = await get_inputs_from_llm()
    logging.info(f"Prompt: {prompt}")

    # diversify the music generation by adding randomness with
    # a high temperature and set a maximum music length
    generate_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 250,
    }

    # Generate the audio clip
    try:
        outputs = music_generator(prompt, generate_kwargs=generate_kwargs)
        audio = outputs["audio"]
        # Flatten the tensor if it's multi-channel to a 2d numpy array
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        sr = outputs["sampling_rate"]
        logging.info(f"Audio: {audio}")
        logging.info(f"Sample rate: {sr}")

    except Exception as e:
        logging.error(f"Exception: {e}")
        st.error("Error while generating audio. Please try again.")
        return None

    # Return the audio Clip
    return (audio, sr)

'''def infer_endpoint(prompt: str = None):
    """
    Function to call the hosted inference endpoint.
    """
    logging.info("Calling the hosted inference endpoint.")
    # If there isn't a prompt, get one from LLM
    if prompt is None:
        logging.debug("No prompt provided. Getting one from LLM.")
        prompt = get_inputs_from_llm()
    url = "https://qoh9yywqip1pjbhd.us-east-1.aws.endpoints.huggingface.cloud"
    inputs = {"inputs": prompt}
    client = InferenceClient(
        model = "https://qoh9yywqip1pjbhd.us-east-1.aws.endpoints.huggingface.cloud",
        token=os.getenv("HUGGINGFACE_API_KEY"))
    logging.debug(f"Sending prompt to inference endpoint: {prompt}")
    response = client.post(json={"inputs": prompt})
    # response looks like this b'[{"generated_text":[[-0.182352,-0.17802449, ...]]}]'
    logging.debug(f"Received response from inference endpoint: {response}")

    output = response.content
    logging.debug(f"Generated audio: {output}")

    return output'''
