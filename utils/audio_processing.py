""" Audio processing, feature extraction, and visualization using torch and librosa"""
from typing import Tuple
from IPython.display import Audio
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Global constants
N_FFT = 1024
WIN_LENGTH = None
HOP_LENGTH = 512

# Set the torchaudio backend to soundfile
torchaudio.set_audio_backend("soundfile")

@st.cache_data
def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """ Load audio from the specified path """
    waveform, sample_rate = torchaudio.load(audio_path)
    # Flatten the tensor if it's multi-channel
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

@st.cache_data
def plot_waveform(_waveform: torch.Tensor, sr: int, title: str = "Waveform") -> go.Figure:
    """ Plot the waveform using plotly """
    num_channels, num_frames = _waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    fig = go.Figure()

    for ch in range(num_channels):
        fig.add_trace(go.Scatter(x=time_axis, y=_waveform[ch], line=dict(width=1), name=f"Channel {ch+1}"))

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
    fig.update_yaxes(gridcolor="gray")
    fig.update_xaxes(gridcolor="gray")
    
    return fig

@st.cache_data
def play_audio(file) -> Audio:
    """ Play the audio waveform using IPython.display.Audio """
    return Audio(filename=file, rate=22050)

@st.cache_data
def get_spectrogram(_waveform):
    """ Get the spectrogram from the waveform """
    n_fft = 1024
    win_length = None
    hop_length = 512

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    return spectrogram(_waveform)

@st.cache_data
def plot_spectrogram(_specgram, title="Spectrogram", ylabel="freq_bin"):
    """ Plot the spectrogram using plotly"""
    # Squeeze the first dimension if it's of size 1
    specgram = _specgram.squeeze(0) if _specgram.shape[0] == 1 else _specgram
    specgram_db = librosa.power_to_db(specgram)
    fig = px.imshow(specgram_db, origin="lower", labels={'x': 'Frame', 'y': ylabel, 'color': 'dB'})
    if title:
        fig.update_layout(title=title)
    
    return fig

@st.cache_data
def generate_mel_spectrogram(_waveform, sample_rate):
    """ Generate the mel spectrogram from the waveform """
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(_waveform)
    melspec = melspec.squeeze(0) if melspec.shape[0] == 1 else melspec

    return melspec

@st.cache_data
def get_mfcc(_waveform, sample_rate):
    """ Get the MFCC from the waveform """
    n_fft = 2048
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )

    mfcc = mfcc_transform(_waveform)

    return mfcc

@st.cache_data
def get_lfcc(_waveform, sample_rate):
    """ Get the LFCC from the waveform """
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_lfcc = 256

    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )

    lfcc = lfcc_transform(_waveform)

    return lfcc

@st.cache_data
def get_pitch(_waveform, sample_rate):
    """ Get the pitch from the waveform """
    pitch = F.detect_pitch_frequency(_waveform, sample_rate)

    return pitch

@st.cache_data
def plot_pitch(_waveform, sr, _pitch):
    """ Plot the waveform and pitch using plotly """
    waveform = _waveform.numpy()
    pitch = _pitch.numpy()
    num_frames = waveform.shape[1]
    end_time = num_frames / sr
    time_axis_waveform = torch.linspace(0, end_time, num_frames)
    time_axis_pitch = torch.linspace(0, end_time, pitch.shape[1])

    # Create the figure with two traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis_waveform, y=waveform[0],
                line=dict(width=1, color="gray"), opacity=0.3, name="Waveform"))
    fig.add_trace(go.Scatter(x=time_axis_pitch, y=pitch[0], line=dict(width=2,
                color="green"), name="Pitch"))

    # Update layout
    fig.update_layout(title="Pitch Feature", xaxis_title="Time", yaxis_title="Amplitude")
    fig.update_yaxes(gridcolor="gray")
    fig.update_xaxes(gridcolor="gray")

    return fig

@st.cache_data
def plot_mfcc(_mfcc):
    """ Plot the MFCC using plotly """
    mfcc = _mfcc
    fig = px.imshow(mfcc.squeeze().numpy(), origin="lower", labels={'x': 'Frame', 'y': 'Coefficients', 'color': 'Value'})
    fig.update_layout(title="MFCC")
    return fig

@st.cache_data  
def plot_lfcc(_lfcc):
    """ Plot the LFCC using plotly """
    lfcc = _lfcc
    fig = px.imshow(lfcc.squeeze().numpy(), origin="lower", labels={'x': 'Frame', 'y': 'Coefficients', 'color': 'Value'})
    fig.update_layout(title="LFCC")
    return fig
