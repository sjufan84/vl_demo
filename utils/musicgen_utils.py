""" Utility functions for the musicgen model """
from typing import List
#from transformers import AutoProcessor, MusicgenForConditionalGeneration
#import streamlit as st
import requests
#from utils.model_utils import get_inputs_from_llm


'''if "llm_inputs" not in st.session_state:
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
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=450)

    # Convert the audio values to a numpy array
    audio = audio_values[0].detach().cpu().numpy()
    
    return audio
'''
""" Utility functions for the musicgen model """
def generate_music(keywords: List[str]):
  API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-stereo-medium"
  headers = {"Authorization": "Bearer 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"}
  payload = {
    "inputs": keywords
  }
  response = requests.post(API_URL, headers=headers, json=payload)
	
  return response.content # Return the audio bytes
