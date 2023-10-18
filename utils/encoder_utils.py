""" Utility functions for encoding audio files using the Encodec model. """
import os
import logging
import torch
from transformers import EncodecModel, AutoProcessor
import numpy as np
import pinecone
from dotenv import load_dotenv
from utils.model_utils import get_similar_audio_clips

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(index_name="musicgen")


# If you want to enable logging
logging.basicConfig(level=logging.INFO)

def chunk_audio(audio_array:np.array=None, sr:int=32000):
    """Chunk the audio file into 15 second chunks"""
    # Check for empty audio data
    if not audio_array.size:
        raise ValueError("Empty audio file")
    audio_chunks = []
    # Process and chunk the audio file
    if len(audio_array) < 15 * sr:
        # Pad the audio data with zeros
        audio_chunks = [np.pad(audio_array, (0, 15 * sr - len(audio_array)), "constant")]
    else:
        audio_chunks = np.array_split(audio_array, len(audio_array) // (15 * sr))
        # If there are any chunks that are less than 15 seconds pad them with zeros
        if len(audio_chunks[-1]) < 15 * sr:
            audio_chunks[-1] = np.pad(audio_chunks[-1],
            (0, 15 * sr - len(audio_chunks[-1])), "constant")

    return audio_chunks

def encode_audio_chunks(audio:np.array=None, sr:int = 32000, model="facebook/encodec_32khz",
                        processor="facebook/encodec_32khz"):
    """Encode the audio chunks using the Encodec model """ 
    # Load the model
    model = EncodecModel.from_pretrained(model)
    # Load the processor
    processor = AutoProcessor.from_pretrained(processor)
    audio_chunks = chunk_audio(audio, sr)
    encoded_audio_chunks = []
    for chunk in audio_chunks:
        inputs = processor(raw_audio=chunk,
        sampling_rate=processor.sampling_rate, return_tensors="pt")
        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        encoded_audio_chunks.append(encoder_outputs.audio_codes.flatten().cpu().numpy())

    # Ensure that the length of the encoded audio chunks is 3000.
    # If it is not, pad it with zeros or slice it to 3000
    if len(encoded_audio_chunks[-1]) < 3000:
        encoded_audio_chunks[-1] = np.pad(encoded_audio_chunks[-1],
        (0, 3000 - len(encoded_audio_chunks[-1])), "constant")
    elif len(encoded_audio_chunks[-1]) > 3000:
        encoded_audio_chunks[-1] = encoded_audio_chunks[-1][:3000]
    
    similar_audio = get_similar_audio_clips(encoded_audio_chunks[-1])
    codes = similar_audio["matches"][0]["values"]
    # Convert the codes to a numpy array
    codes_np = np.array(codes)
    # Reshape the array to the correct shape
    original_shape = (1, 1, 4, 750)
    retrieved_audio_codes = np.reshape(codes_np, original_shape)

    # Convert the retrieved audio codes to a tensor
    codes_int = torch.tensor(retrieved_audio_codes).int()

    # Step 4: Decode
    decoded_audio = model.decode(
        audio_codes=codes_int,
        audio_scales=[None]  # Assuming this tensor is available from your encoding step
        # Assuming this tensor is available from your encoding step
    )  # Assuming you are interested in `audio_values`

    # Convert the tensor to a numpy array
    decoded = decoded_audio[0].cpu().detach().numpy()

    # Flatten the numpy array to 2d
    decoded = decoded.reshape(-1)
    
    return decoded
