""" Audio processing utilities live here. """

import os
from dotenv import load_dotenv
import numpy as np
import librosa
import librosa.display
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, DCT, Deltas, ContextWindow, InputNormalization 
import torch
from sklearn.manifold import TSNE
import streamlit as st
# Load the environment variables
load_dotenv()

# Load the Hugging Face API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_KEY")



def get_waveform_data(wav_file):    
    """ Loads the audio data from librosa
    and displays the waveform. """
    audio, sr = librosa.load(wav_file, sr=None)

    # Return a tuple of the audio data and the sample rate
    return audio, sr

def extract_features(file_name):
    """ Extracts the features from the audio file. """
    signal = read_audio(file_name)
    signal = signal.unsqueeze(0)
    compute_STFT = STFT(
    sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    )
    features = compute_STFT(signal)
    features = spectral_magnitude(features)
    compute_fbanks = Filterbank(n_mels=40)
    features = compute_fbanks(features)
    compute_mfccs = DCT(input_size=40, n_out=20)
    features = compute_mfccs(features)
    compute_deltas = Deltas(input_size=20)
    delta1 = compute_deltas(features)
    delta2 = compute_deltas(delta1)
    features = torch.cat([features, delta1, delta2], dim=2)
    compute_cw = ContextWindow(left_frames=5, right_frames=5)
    features  = compute_cw(features)
    norm = InputNormalization()
    features = norm(features, torch.tensor([1]).float())

    return features

def perform_tsne():

    # Step 1: Extract features for all files
    all_features = []
    for file in st.session_state.audio_files.keys():
        feature = extract_features(st.session_state.audio_files[file])
        st.write(feature.shape)
        all_features.append(feature)
    # Step 2: Concatenate all features into one array   
    combined_features = np.concatenate(all_features, axis=0)

    # Step 3: Perform TSNE on the combined array
    tsne = TSNE(n_components=2)  # or however many dimensions you want
    tsne_results = tsne.fit_transform(combined_features)

    # Step 4: Split the results back out
    split_points = np.cumsum([features.shape[0] for features in all_features])
    split_tsne_results = np.split(tsne_results, split_points[:-1])

    # Step 5: Save each result to the session state
    for i, file in enumerate(st.session_state.audio_files.keys()):
        st.write(f"{file}: {split_tsne_results[i]}")
        st.session_state.tsne_results[file] = split_tsne_results[i].tolist()
    return split_tsne_results
   
def get_embeddings(wav_file):
    """ Extracts the embeddings from the audio file. """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    signal, fs =torchaudio.load(wav_file)

    # Compute speaker embeddings
    embeddings = classifier.encode_batch(signal)

    # Average over time dimension to get a single vector per audio clip
    averaged_embeddings = embeddings.mean(axis=1)

    return averaged_embeddings.cpu().numpy()
