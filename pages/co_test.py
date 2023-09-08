import streamlit as st
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
from scipy.spatial.distance import cosine
import os
import pickle  # For storing pre-encoded songs

# Function to load and encode audio
def load_and_encode_audio(audio_file_path, model):
    wav, sr = torchaudio.load(audio_file_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    return codes

# Load pre-encoded artist songs from a pickle file
try:
    with open("encoded_artist_songs.pkl", "rb") as f:
        artist_encoded_songs = pickle.load(f)
except FileNotFoundError:
    artist_encoded_songs = {}

# Streamlit app
st.title("Audio Similarity App")

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Upload user's audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Encode user's uploaded audio
    with open("temp_audio_file.mp3", "wb") as f:
        f.write(uploaded_file.read())
    
    user_song_codes = load_and_encode_audio("temp_audio_file.mp3", model)
    os.remove("temp_audio_file.mp3")  # Delete the temporary file

    # Placeholder for similarity scores
    similarity_scores = {}

    # Compute similarity between user song and each artist song
    for song, artist_codes in artist_encoded_songs.items():
        similarity_score = 1 - cosine(user_song_codes.flatten(), artist_codes.flatten())
        similarity_scores[song] = similarity_score

    # Find the most similar song
    most_similar_song = max(similarity_scores, key=similarity_scores.get)
    st.write(f"The most similar song by the artist is: {most_similar_song}")

def encode_file_batch():
    # Placeholder for storing encoded songs
    artist_encoded_songs = {}

    # Loop through the artist's songs
    for song_file in os.listdir("./audio_files/artist_main"):
        song_codes = load_and_encode_audio(f"./audio_files/artist_main/{song_file}", model)
        artist_encoded_songs[song_file]["codes"] = song_codes
        artist_encoded_songs[song_file]["artist"] = "Luke Combs"  # Change as needed
        # Isolate the song name
        song_name = song_file.split(".")[0]
        artist_encoded_songs[song_file]["song_name"] = song_name

    # Save the dictionary to a pickle file
    with open("encoded_artist_songs.pkl", "wb") as f:
        pickle.dump(artist_encoded_songs, f)

# Encode the artist's songs
encode_file_batch()
