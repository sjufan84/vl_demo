""" Utility functions for the musicgen model """
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from utils.model_utils import get_inputs_from_llm

async def load_models():
    """ Load the models """
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

    return processor, model

async def get_music(array: np.array, text: str = None,
        sr: int = 32000, model = None, processor = None):
    """ Get the music sample from the musicgen model """
    processor, model = await load_models()
    # Divide the array by 2
    array = array / 2
    
    if text:
        text = [text]
    else:
        text = [get_inputs_from_llm()]

    inputs = processor(
        audio=array,
        sampling_rate=sr,
        text=text,
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

    return audio_values