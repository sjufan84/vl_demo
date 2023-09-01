import soundfile as sf
from transformers import EncodecModel, AutoProcessor
import logging

def chunk_and_encode_encodec(audio_file_path, model_name="facebook/encodec_48khz", processor_name="facebook/encodec_48khz"):
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
    
    # Assuming mono audio, reshape if necessary
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    # Process and chunk the audio file
    chunk_size = int(sr * 15)  # 15 seconds, adjust as needed
    n_chunks = len(audio_data) // chunk_size
    encoded_chunks = [None] * n_chunks
    
    for i in range(0, len(audio_data), chunk_size):
        logging.info(f"Processing chunk {i // chunk_size + 1}/{n_chunks}")
        
        chunk = audio_data[i:i + chunk_size]
        
        print("Chunk shape before processing:", chunk.shape)  # Debugging line
        
        inputs = processor(raw_audio=chunk, sampling_rate=sr, return_tensors="pt")
        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        
        encoded_chunks[i // chunk_size] = encoder_outputs.audio_codes
        
    return encoded_chunks

# If you want to enable logging
logging.basicConfig(level=logging.INFO)

