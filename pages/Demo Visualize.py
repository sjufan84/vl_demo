""" Demo page for visualizing audio features """
import tempfile
import base64   
import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import librosa
import numpy as np
import soundfile as sf
import streamlit.components.v1 as components
from utils.audio_processing import (
    load_audio, get_spectrogram, get_mfcc, get_lfcc, get_pitch,
    plot_waveform, plot_spectrogram, plot_mfcc, plot_lfcc, plot_pitch
)


def audio_player_component():
    # specify directory and initialize st_audiorec object functionality
    audio_player = components.declare_component("audio_player", url="http://localhost:3001")

    return audio_player


def demo_visualize_page():
    """ Demo page for visualizing audio features """
    audio_files = ['./audio_samples/avicii1.wav', './audio_samples/combs1.wav']
    labels = ["Aloe Blacc", "Luke Combs"]  # Labels for the audio clips
    
    # Load and process audio files
    audio_waveforms = [load_audio(file) for file in audio_files]

    # Select feature to visualize
    feature_option = st.selectbox(
        "Select a feature to visualize:",
        ("Waveform", "Spectrogram", "Mel Spectrogram", "MFCC", "LFCC", "Pitch", "KMeans Clustering")
    )

    for i, (waveform, sr) in enumerate(audio_waveforms):
        plot_feature(waveform, sr, feature_option, audio_files[i], labels[i])


def plot_feature(waveform, sr, feature_option, audio_file, label):
    """ Plot the selected feature """
    col1, col2 = st.columns([2, 1])  # Adjust ratios as needed
    with col1:
        st.markdown(f"**{label} - {feature_option}**")  # Add the label and title
        if feature_option == "Waveform":
            col1.plotly_chart(plot_waveform(waveform, sr))
        elif feature_option == "Spectrogram":
            col1.plotly_chart(plot_spectrogram(get_spectrogram(waveform)))
        # ... (rest of the code remains the same)
    with col2:
        st.markdown(f"**{label} - Audio Clip**")  # Add the label for the audio clip
        col2.audio(audio_file, format='audio/wav')

def read_audio(file_path):
    signal, _ = librosa.load(file_path, sr=16000)
    return signal

def compute_STFT(signal, n_fft=400, hop_length=10, win_length=25):
    return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def compute_ISTFT(stft_matrix, hop_length=10, win_length=25):
    return librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length)

# Create temporary audio files for the segments
def create_audio_files(segments, sr=16000):
    files = []
    for segment in segments:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, segment, samplerate=sr)
        files.append(tmp_file)
    return files

# Create a data URL for an audio file
def audio_file_to_data_url(file):
    audio_encoded = base64.b64encode(file.read())
    return "data:audio/wav;base64," + audio_encoded.decode()

def extract_features(signal):
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=13).T
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=16000).T
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=signal, sr=16000).T
    # Concatenate all the features (vertically)
    return np.hstack([mfccs, contrast, chroma])
# Function to create audio data URLs

def create_audio_urls(segments):
    urls = []
    for segment in segments:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        librosa.output.write_wav(tmp_file.name, segment, sr=16000)
        with open(tmp_file.name, "rb") as file:
            audio_encoded = base64.b64encode(file.read())
        urls.append("data:audio/wav;base64," + audio_encoded.decode())
    return urls

def test_plots():

    st.markdown("""
    ### Melodic Voiceprint: A Harmony of Science, Art, and Security

    The 3D chart visualizes the unique features that compose the voice of two different artists.
    Each point represents a segment of a song, and the position of the points reflects various
    characteristics of the voice such as pitch, rhythm, and timbre.
    The segments are grouped by color, highlighting similarities and differences between the artists.
    """)

    # Read and preprocess audio signals
    avicii_signal = read_audio('./audio_samples/avicii1.wav')
    combs_signal = read_audio('./audio_samples/combs1.wav')
    min_length = min(len(avicii_signal), len(combs_signal))
    avicii_signal = avicii_signal[:min_length]
    combs_signal = combs_signal[:min_length]
    

    # Extract features
    avicii_features = extract_features(avicii_signal)
    combs_features = extract_features(combs_signal)

    # Transpose to have features as columns
    avicii_features = avicii_features.T
    combs_features = combs_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(avicii_features), pd.DataFrame(combs_features)], axis=0)

    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=40)
    kmeans.fit(pca_df)
    clusters = kmeans.predict(pca_df)

    # Create a DataFrame for plotting
    cluster_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])
    cluster_df['cluster'] = clusters

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    avicii_segments = np.array_split(avicii_signal, len(cluster_df)//2)
    combs_segments = np.array_split(combs_signal, len(cluster_df)//2)

   # Create labels for the segments
    avicii_labels = [f"Avicii - Segment {i+1}" for i in range(len(avicii_segments))]
    combs_labels = [f"Luke Combs - Segment {i+1}" for i in range(len(combs_segments))]
    segment_labels = avicii_labels + combs_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(avicii_segments))] * 2

    # Add segment names and numbers to the DataFrame
    cluster_df['segment_name'] = segment_labels
    cluster_df['segment_number'] = segment_numbers


    
    # Create temporary audio files
    avicii_files = create_audio_files(avicii_segments)
    combs_files = create_audio_files(combs_segments)

    # Create audio URLs
    avicii_audio_urls = [audio_file_to_data_url(file) for file in avicii_files]
    combs_audio_urls = [audio_file_to_data_url(file) for file in combs_files]
    audio_urls = avicii_audio_urls + combs_audio_urls

    fig = px.scatter_3d(
    cluster_df,
    x='PC1',
    y='PC2',
    z='PC3',
    color='segment_number',
    color_continuous_scale='rainbow',
    title='3d Representation of Vocal Features (Aloe Blacc vs. Luke Combs)',
    text='segment_name',  # Label the points with segment names
    )
    fig.update_layout(
        width=750,  # Width of the plot in pixels
        height=750,  # Height of the plot in pixels
        scene=dict(
            xaxis=dict(title='PC1'),
            yaxis=dict(title='PC2'),
            zaxis=dict(title='PC3')
        ),
        showlegend=False,
    )
    fig.update_traces(
    textposition='top center',  # Position of the text labels
    textfont_size=10,
    marker_size=8            # Font size of the text labels
)
    col1, col2, col3 = st.columns([4, 0.25, 1], gap="small")
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.text("")
    with col3:
        # Create a dropdown list for selecting a segment
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        st.success("""**To drill down even further, select a segment to play the audio clip\
                    that corresponds to that segment.**""")
        selected_segment = st.selectbox("Select a Segment to Play:", segment_labels)

        # Get the index of the selected label in the segment_labels list
        selected_index = segment_labels.index(selected_segment)

        # Render the custom component for audio playback for the selected segment
        audio_player = audio_player_component()
        audio_player(audioUrls=[audio_urls[selected_index]], segmentNames=[selected_segment])

    st.markdown("""
                 **So What Does This Mean for Music, Security, and the Future of the Industry?**

    1. **Understanding the Voice**: By analyzing these features, we can create a "Melodic Voiceprint,"
                 a unique signature of an artist's voice. It's like a fingerprint for their voice, capturing the subtle nuances that make their voice distinctly theirs.

    2. **Protecting Authenticity**: The Melodic Voiceprint can be used to determine whether a piece of
        audio is genuinely from the claimed artist or not. It's a powerful tool to detect deepfakes,
        which are artificially created audio files that convincingly imitate a real artist's voice.

    3. **Application in Music**: For musicians, the Melodic Voiceprint
        safeguards artistic integrity. It ensures that their creative work
        remains authentic and unaltered, protecting against potential deepfake manipulation.

    4. **A New Layer of Security**: In the digital age, where voices can be forged,
        the Melodic Voiceprint acts as a cutting-edge solution to maintain the authenticity of vocal identity.
                
    5. **Downstream Possibilities**:
    * **Content Generation**: The Melodic Voiceprint can be utilized to develop personalized content, such as custom music, voiceovers, and more.
    * **Voice Authentication**: It offers a robust method for voice-based authentication in security systems.
    * **Enhanced Creativity**: Musicians and creators can experiment with voice manipulation, remixing, and other artistic expressions while preserving authenticity.
    * **New Business Models**: The Melodic Voiceprint can be used to create new revenue streams for artists, such as personalized content and voice authentication.
    """)

    

test_plots()      