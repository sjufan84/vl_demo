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
    signal, fs =torchaudio.load(file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)  # ensure signal is on the same device as the model
    window_length = int(fs * 1.0)  # 1 second window
    overlap = int(fs * 0.5)  # 50% overlap
    embeddings = []
    for i in range(0, len(signal), overlap):
        window = signal[i:i+window_length]
        if len(window) == window_length:
            embedding = classifier.encode_batch(window)
            embeddings.append(embedding)
    # Convert the list of embeddings to a 2D array
    return embeddings




def perform_kmeans_clustering(data, n_clusters=2):
    """ Performs Kmeans clustering on the data. """
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    return kmeans.labels_

def perform_tsne(features, n_components=2):
    """ Performs t-SNE on the data. """
    tsne = TSNE(n_components=n_components)
    tsne_result = tsne.fit_transform(features)
    return tsne_result

