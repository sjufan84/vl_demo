""" Audio processing utilities live here. """

import os
from dotenv import load_dotenv
import numpy as np
import librosa
import librosa.display
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, Filterbank, DCT, Deltas, ContextWindow, InputNormalization 
from sklearn.manifold import TSNE
import streamlit as st
# Load the environment variables
load_dotenv()

# Load the Hugging Face API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_KEY")

def get_spectrogram(stft):
    """ Generates the spectrogram from the audio file. """
    #signal = process_audio(wav_file)
    #signal_stft = compute_stft(signal)
    spectrogram = stft.pow(2).sum(-1) # Power spectrogram
    spectrogram = spectrogram.squeeze(0).transpose(0,1)
    spectrogram = torch.log(spectrogram)
    return spectrogram

def process_audio(wav_file):
    """ Reads in an audio file and returns the signal. """
    signal = read_audio(wav_file)
    processed_signal = signal.unsqueeze(0)
    return processed_signal

def get_waveform_data(wav_file):    
    """ Loads the audio data from librosa
    and displays the waveform. """
    audio, sr = librosa.load(wav_file, sr=None)

    # Return a tuple of the audio data and the sample rate
    return audio, sr

def compute_stft(signal):
    """ Compute the short-time Fourier transform from the audio file. """
    stft = STFT(
    sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    )
    return stft(signal)    

def compute_fbanks(stft):
    """ Compute the filterbanks from the STFT. """
    fbanks = Filterbank(n_mels=40)
    return fbanks(stft)

def compute_mfccs(fbanks):
    """ Compute the MFCCs from the filterbanks. """
    mfccs = DCT(input_size=40, n_out=20)
    return mfccs(fbanks)

def compute_delta1(mfccs):
    """ Compute the deltas from the MFCCs. """
    deltas = Deltas(input_size=20)
    return deltas(mfccs)

def compute_delta2(delta1):
    """ Compute the deltas from the MFCCs. """
    deltas = Deltas(input_size=20)
    return deltas(delta1)

def compute_cw(mfccs, delta1, delta2):
    """ Compute the context window from the MFCCs. """
    cw = ContextWindow(left_frames=5, right_frames=5)
    features = torch.cat([mfccs, delta1, delta2], dim=2)
    return cw(features)

def normalize_features(cw):
    """ Normalize the features. """
    norm = InputNormalization()
    return norm(cw, torch.tensor([1]).float())  

def extract_norm_features(file_name):
    """ Extracts the features from the audio file. """
    signal = process_audio(file_name)
    stft = compute_stft(signal)
    fbanks = compute_fbanks(stft)
    mfccs = compute_mfccs(fbanks)
    delta1 = compute_delta1(mfccs)
    delta2 = compute_delta2(delta1)
    cw = compute_cw(mfccs, delta1, delta2)
    norm = normalize_features(cw)
    return {"signal": signal, "stft": stft, "fbanks": fbanks, "mfccs": mfccs, "delta1": delta1, "delta2": delta2, "cw": cw, "norm": norm}

def perform_tsne():
    """ Performs TSNE on the normalized features. """

    # Step 1: Extract features for all files
    all_features = []
    for file in st.session_state.audio_files.keys():
        feature = extract_norm_features(st.session_state.audio_files[file])
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
    signal, fs = torchaudio.load(wav_file)

    # Compute speaker embeddings
    embeddings = classifier.encode_batch(signal)

    # Average over time dimension to get a single vector per audio clip
    averaged_embeddings = embeddings.mean(axis=1)

    return averaged_embeddings.cpu().numpy()
