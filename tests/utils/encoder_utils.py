""" Utility functions for encoding audio files using the Encodec model. """
import os
from pathlib import Path
import numpy as np
import soundfile as sf
from transformers import EncodecModel, AutoProcessor
import pandas as pd
from sklearn.decomposition import PCA
import logging


# If you want to enable logging
logging.basicConfig(level=logging.INFO)

def chunk_and_encode_encodec(audio_file_path, model_name="facebook/encodec_48khz",
                            processor_name="facebook/encodec_48khz"):
    """Chunk and encode an audio file using the Encodec model."""
    # Initialize the Encodec model and processor
    model = EncodecModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(processor_name)
    
    print("Processor feature size:", processor.feature_size)  # Debugging line
    
    # Load the audio file
    audio_data, sr = sf.read(audio_file_path)
    
    # Validate sample rate
    if sr != 48000:
        raise ValueError(f"Unexpected sample rate: {sr}")
    
    # Check for empty audio data
    if not audio_data.size:
        raise ValueError("Empty audio file")
    
    # Check if the audio is stereo and pass as is
    if len(audio_data.shape) == 1 or audio_data.shape[1] != 2:
        raise ValueError("The model expects stereo audio.")
        
    # Process and chunk the audio file
    chunk_size = int(sr * 15)  # 15 seconds, adjust as needed
    n_chunks = len(audio_data) // chunk_size
    encoded_chunks = [None] * n_chunks
    
    for i in range(0, len(audio_data), chunk_size):
        logging.info(f"Processing chunk {i // chunk_size + 1}/{n_chunks}")

        chunk = audio_data[i:i + chunk_size]  # This line should come before you use 'chunk'

        # Transpose the dimensions so that channels come first, followed by length
        chunk = np.transpose(chunk, (1, 0))

        print("Chunk shape after transposing:", chunk.shape)  # Debugging line

        # Your existing code for processing the chunk
        inputs = processor(raw_audio=chunk, sampling_rate=sr, return_tensors="pt")

        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        
        encoded_chunks[i // chunk_size] = encoder_outputs.audio_codes


def encode_pca_file_folder(folder_path):
    """ Encode all audio files in a folder using PCA."""
    pca = PCA(n_components=1536)  # Adjust as needed
    # Initialize a DataFrame to hold all the data
    df = pd.DataFrame()
    # Set the audio files to be all files in the folder
    audio_files = os.listdir(folder_path)
    # Iterate through each audio file and apply the function
    for audio_file in audio_files:
        full_path = os.path.join(folder_path, audio_file)
        try:
            # Extract and encode audio chunks
            encoded_chunks = chunk_and_encode_encodec(full_path)
            
            # Apply PCA to each chunk
            reduced_chunks = [pca.transform(chunk) for chunk in encoded_chunks]
            
            # Create a DataFrame for this audio file
            audio_df = pd.DataFrame({
                'segment': range(len(reduced_chunks)),
                'reduced_audio': reduced_chunks,
                'song': [audio_file]*len(reduced_chunks)
            })
            
            # Append to the master DataFrame
            df = pd.concat([df, audio_df], ignore_index=True)
            
            print(f"Successfully processed {audio_file}")
            
        except Exception as e:
            print(f"Failed to process {audio_file}: {e}")

    return df
