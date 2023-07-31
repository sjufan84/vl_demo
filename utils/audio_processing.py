""" Audio processing utilities live here. """

import os
from dotenv import load_dotenv
import numpy as np
import librosa
import librosa.display
import torchaudio
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from speechbrain.pretrained import EncoderClassifier
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

import torch

def extract_features(file_name):
    """ Extracts the features from the audio file. """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    classifier.hparams.label_encoder.ignore_len()
    signal, fs = torchaudio.load(file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)  # ensure signal is on the same device as the model
    embeddings = classifier.encode_batch(signal)
    # Convert the tensor to numpy and reshape it
    embeddings = embeddings.cpu().numpy().reshape(1, -1)
    return embeddings








def perform_kmeans_clustering(data, n_clusters=2):
    """ Performs Kmeans clustering on the data. """
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    return kmeans.labels_


def perform_tsne(n_components=2):

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
    st.write(tsne_results)

    # Step 4: Split the results back out
    split_points = np.cumsum([features.shape[0] for features in all_features])
    split_tsne_results = np.split(tsne_results, split_points[:-1])

    # Step 5: Save each result to the session state
    # Step 5: Save each result to the session state
    for i, file in enumerate(st.session_state.audio_files.keys()):
        st.write(f"{file}: {split_tsne_results[i]}")
        st.session_state.tsne_results[file] = split_tsne_results[i].tolist()
    return split_tsne_results



