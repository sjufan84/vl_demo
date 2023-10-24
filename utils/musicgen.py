""" Utility functions for the musicgen model """
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import streamlit as st
from utils.model_utils import get_inputs_from_llm

if "llm_inputs" not in st.session_state:
    st.session_state.llm_inputs = None

@st.cache_resource
def load_models():
    """ Load the models """
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    return processor, model

async def get_music(text: str = st.session_state.llm_inputs,
    sr: int = 32000, model = None, processor = None):
    """ Get the music sample from the musicgen model """
    processor, model = load_models()
    # Divide the array into chunks of 6 seconds or less
    #if array.shape[0] > 192000:
    #    array = array[:192000]

    if text:
        text = [text]
    else:
        text = [get_inputs_from_llm()]
    inputs = processor(
        #audio=array,
        sampling_rate=sr,
        text=text,
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=750)

    # Convert the audio values to a numpy array
    audio = audio_values[0].detach().cpu().numpy()
    
    return audio

async def hf_inference_api(texts)
    import json
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
data = query("Can you please let us know more details about your ")