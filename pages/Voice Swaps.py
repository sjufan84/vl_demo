""" Demo page for visualizing audio features """ 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import librosa
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Voice Lockr Demo", page_icon=":microphone:",
                initial_sidebar_state="collapsed", layout="wide")

def read_audio(file_path):
    """ Read an audio file and return the signal """
    signal, _ = librosa.load(file_path, sr=16000)
    return signal

def extract_features(signal):
    """ Extract features from an audio signal """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=13).T
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=16000).T
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=signal, sr=16000).T
    # Concatenate all the features (vertically)
    return np.hstack([mfccs, contrast, chroma])


@st.cache
def get_3d_chart_ghost():
    """ Get the 3D chart for Fast Car """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/jenny_ghost.wav')
    lc_signal = read_audio('./audio_samples/jenny_ghost.wav')
    joel_signal = read_audio('./audio_samples/joel_ghost.wav')
    min_length = min(len(jenny_signal), len(lc_signal), len(joel_signal))
    jenny_signal = jenny_signal[:min_length]
    lc_signal = lc_signal[:min_length]
    joel_signal = joel_signal[:min_length]
    

    # Extract features
    jenny_features = extract_features(jenny_signal)
    lc_features = extract_features(lc_signal)
    joel_features = extract_features(joel_signal)

    # Transpose to have features as columns
    jenny_features = jenny_features.T
    lc_features = lc_features.T
    joel_features = joel_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(jenny_features), pd.DataFrame(lc_features), pd.DataFrame(joel_features)], axis=0)
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])


    # Create segments (you may need to adjust this to match your actual segmentation logic)
    jenny_segments = np.array_split(jenny_signal, len(plot_df)//3)
    lc_segments = np.array_split(lc_signal, len(plot_df)//3)
    joel_segments = np.array_split(joel_signal, len(plot_df)//3)

   # Create labels for the segments
    jenny_labels = [f"Jenny - Segment {i+1}" for i in range(len(jenny_segments))]
    lc_labels = [f"LC - Segment {i+1}" for i in range(len(lc_segments))]
    joel_labels = [f"Joel - Segment {i+1}" for i in range(len(joel_segments))]
    segment_labels = jenny_labels + lc_labels + joel_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(lc_segments))] * 3

    # Add segment names and numbers to the DataFrame
    plot_df['segment_name'] = segment_labels
    plot_df['segment_number'] = segment_numbers
    
    col1, col2 = st.columns([1.75, 1], gap='large')


    with col2:# Add a Streamlit multiselect widget to allow users to select artists
        st.text("")
        st.text("")
        st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa
       
        st.markdown("**Original Audio Clips:**")
        joel_bytes = librosa.to_mono(joel_signal)
        lc_bytes = librosa.to_mono(lc_signal)
        jenny_bytes = librosa.to_mono(jenny_signal)
        st.markdown("**LC Fast Car**")
        st.audio(lc_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Joel Fast Car**")
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Jenny Fast Car**")
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        options=['Jenny', 'LC', 'Joel'],
        default=['Jenny', 'LC', 'Joel'],
        )
        # Filter the DataFrame based on selected artists
        filtered_plot_df = plot_df[plot_df['segment_name'].str.contains('|'.join(selected_artists))]


    with col1:# Plot using the filtered DataFrame
        fig = px.scatter_3d(
        filtered_plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Vocal Features -- LC, Joel, Jenny',
        text='segment_name',
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
        st.plotly_chart(fig, use_container_width=True)
            
def get_3d_chart_fcar():
    """ Get the 3D chart for Fast Car """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/lc_fcar.wav')
    lc_signal = read_audio('./audio_samples/jenny_fcar.wav')
    joel_signal = read_audio('./audio_samples/joel_fcar.wav')
    min_length = min(len(jenny_signal), len(lc_signal), len(joel_signal))
    jenny_signal = jenny_signal[:min_length]
    lc_signal = lc_signal[:min_length]
    joel_signal = joel_signal[:min_length]
    
    # Extract features
    jenny_features = extract_features(jenny_signal)
    lc_features = extract_features(lc_signal)
    joel_features = extract_features(joel_signal)

    # Transpose to have features as columns
    jenny_features = jenny_features.T
    lc_features = lc_features.T
    joel_features = joel_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(jenny_features), pd.DataFrame(lc_features), pd.DataFrame(joel_features)], axis=0)
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    jenny_segments = np.array_split(jenny_signal, len(plot_df)//3)
    lc_segments = np.array_split(lc_signal, len(plot_df)//3)
    joel_segments = np.array_split(joel_signal, len(plot_df)//3)

   # Create labels for the segments
    jenny_labels = [f"Jenny - Segment {i+1}" for i in range(len(jenny_segments))]
    lc_labels = [f"LC - Segment {i+1}" for i in range(len(lc_segments))]
    joel_labels = [f"Joel - Segment {i+1}" for i in range(len(joel_segments))]
    segment_labels = jenny_labels + lc_labels + joel_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(lc_segments))] * 3

    # Add segment names and numbers to the DataFrame
    plot_df['segment_name'] = segment_labels
    plot_df['segment_number'] = segment_numbers

    col1, col2 = st.columns([1.75, 1], gap='large') # Create two columns, one for the chart and one for the audio

    with col2:  # Add a Streamlit multiselect widget to allow users to select artists
        st.text("")
        st.text("")
        st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa
       
        st.markdown("**Original Audio Clips:**")
        joel_bytes = librosa.to_mono(joel_signal)
        lc_bytes = librosa.to_mono(lc_signal)
        jenny_bytes = librosa.to_mono(jenny_signal)
        st.markdown("**LC Fast Car**")
        st.audio(lc_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Joel Fast Car**")
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Jenny Fast Car**")
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        options=['Jenny', 'LC', 'Joel'],
        default=['Jenny', 'LC', 'Joel'],
        )
        # Filter the DataFrame based on selected artists
        filtered_plot_df = plot_df[plot_df['segment_name'].str.contains('|'.join(selected_artists))]


    with col1:  # Plot using the filtered DataFrame
        fig = px.scatter_3d(
        filtered_plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Vocal Features -- LC, Joel, Jenny',
        text='segment_name',
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
        st.plotly_chart(fig, use_container_width=True)

def voice_swap_home():
    """ Home page for voice swap visuals """
    st.markdown("""
                ##### :blue[Melodic Voiceprint: A Harmony of Science, Art, and Security]
                
                So what does this voice swap look like in practice?  Below are two examples of voice swaps
                that were generated using the same model that was trained on Jenny and Joel's voices.
                We have two examples: 1) a voice swap of Jenny and Joel's voice singing a clip from Luke
                Combs's "Fast Car", and 2) a voice swap of Jenny and Joel's voice singing a clip from
                Ella Henderson's "Ghost".  While it might seem difficult to swap a male voice to a female voice,
                or vice versa, in reality it is not too much more difficult than like to like, thus further
                widening the scope of the problem.""")