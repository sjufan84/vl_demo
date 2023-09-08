""" Utility functions for encoding audio files using the Encodec model. """
import os
import logging
from io import BytesIO
from transformers import EncodecModel, AutoProcessor
import pandas as pd
from sklearn.decomposition import PCA
import librosa


# If you want to enable logging
logging.basicConfig(level=logging.INFO)

def chunk_and_encode_encodec(audio_file_path=None, audio_bytes=None,
                            model_name="facebook/encodec_24khz",
                            processor_name="facebook/encodec_24khz"):
    """Chunk and encode an audio file or bytes using the Encodec model."""
    # Initialize the Encodec model and processor
    model = EncodecModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(processor_name)
    
    # Determine the source of the audio: file path or bytes
    if audio_file_path:
        audio_data, sr = librosa.load(audio_file_path, mono=True, sr=24000)
    elif audio_bytes:
        audio_data, sr = librosa.load(BytesIO(audio_bytes), mono=True, sr=24000)
    else:
        raise ValueError("Either audio_file_path or audio_bytes must be provided.")
    
    # Check for empty audio data
    if not audio_data.size:
        raise ValueError("Empty audio file")

    # Process and chunk the audio file
    chunk_size = int(sr * 15)  # 15 seconds, adjust as needed
    n_chunks = len(audio_data) // chunk_size
    # If n_chunks <= 1, pad the audio data
    if n_chunks <= 1:
        n_chunks = 1
        audio_data = librosa.util.pad_center(data=audio_data, size=chunk_size)

    encoded_chunks = [None] * n_chunks
    
    for i in range(0, len(audio_data), chunk_size):
        chunk_index = i // chunk_size
        logging.info(f"Processing chunk {chunk_index + 1}/{n_chunks}")

        chunk = audio_data[i:i + chunk_size]  # This line should come before you use 'chunk'
        
        print(f"Chunk shape before transposing: {chunk.shape}")  # Debugging line

        # If the channels are not in the first dimension, you can transpose
        chunk = chunk.T

        print("Chunk shape after transposing:", chunk.shape)  # Debugging line

        # Your existing code for processing the chunk
        inputs = processor(raw_audio=chunk, sampling_rate=sr, return_tensors="pt")

        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        
        # encoded_chunks[i // chunk_size] = encoder_outputs.audio_codes

        if chunk_index < n_chunks:
            encoded_chunks[chunk_index] = encoder_outputs.audio_codes
        else:
            # Handle partial chunk
            logging.warning(f"Partial chunk detected. Ignored.")

    return encoded_chunks

def encode_pca_file_folder(folder_path):
    """ Encode all audio files in a folder using PCA."""
    # Initialize a list to collect all encoded chunks
    all_encoded_chunks = []
    
    # Initialize a DataFrame to hold all the data
    df = pd.DataFrame()
    
    # Set the audio files to be all files in the folder
    audio_files = os.listdir(folder_path)
    
    # First Pass: Collect all encoded chunks
    for audio_file in audio_files:
        full_path = os.path.join(folder_path, audio_file)
        try:
            # Extract and encode audio chunks
            encoded_chunks = chunk_and_encode_encodec(full_path)
            all_encoded_chunks.extend(encoded_chunks)
            
        except Exception as e:
            print(f"Failed to process {audio_file}: {e}")
    
    # Fit PCA model
    pca = PCA(n_components=1536)  # Adjust as needed
    pca.fit(all_encoded_chunks)
    
    # Second Pass: Apply PCA and collect data
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

def encode_and_upsert(folder_path):
    """ Encode all audio files in a folder and upsert to a vector database."""
    # Initialize a DataFrame to hold all the data
    df = pd.DataFrame()

    # Set the audio files to be all files in the folder
    audio_files = os.listdir(folder_path)

    # Expected dimensionality (replace this with the actual dimensionality you expect)
    expected_dim = 2250

    # Iterate through each audio file and apply the function
    for audio_file in audio_files:
        full_path = os.path.join(folder_path, audio_file)
        try:
            # Extract and encode audio chunks
            encoded_chunks = chunk_and_encode_encodec(full_path)  # Assume this function returns encoded chunks
                        
            if encoded_chunks is None or not isinstance(encoded_chunks, (list, tuple)):
                raise ValueError(f"Invalid or None value returned for encoded_chunks for {audio_file}")
            
            # Flatten and validate each chunk
            flat_chunks = []
            for chunk in encoded_chunks:
                flat_chunk = chunk.flatten().cpu().numpy()
                
                if flat_chunk.size != expected_dim:
                    logging.warning(f"Skipping chunk due to unexpected dimensionality: {flat_chunk.size}")
                    continue
                
                flat_chunks.append(flat_chunk)

            # Create a DataFrame for this audio file
            audio_df = pd.DataFrame({
                'segment': range(len(flat_chunks)),
                'encoded_audio': flat_chunks,
                'song': [audio_file] * len(flat_chunks)
            })

            # Append to the master DataFrame
            df = pd.concat([df, audio_df], ignore_index=True)
            
            print(f"Successfully processed {audio_file}")

        except Exception as e:
            print(f"Failed to process {audio_file}: {e}")

    # At this point, `df` contains the flattened, same-dimensionality encoded audio for each segment
    # You can now upsert this into your vector database

    return df

def flatten_chunks(encoded_chunks):
    """ Encode all audio files in a folder and upsert to a vector database."""
    # Expected dimensionality (replace this with the actual dimensionality you expect)
    expected_dim = 2250

    # Flatten and validate each chunk
    flat_chunks = []
    for chunk in encoded_chunks:
        flat_chunk = chunk.flatten().cpu().numpy()
        
        if flat_chunk.size != expected_dim:
            logging.warning(f"Skipping chunk due to unexpected dimensionality: {flat_chunk.size}")
            continue
        
        flat_chunks.append(flat_chunk)
    
    return flat_chunks
