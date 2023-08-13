""" Demo page for visualizing audio features """
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from utils.audio_processing import (
    load_audio, plot_waveform, get_spectrogram, plot_spectrogram,
    generate_mel_spectrogram, get_mfcc, get_lfcc, get_pitch, plot_pitch,
    plot_mfcc, plot_lfcc
)

# Function for KMeans clustering
def plot_results(pca_df):
    """ Perform KMeans clustering on the PCA features """
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(pca_df)
    clusters = kmeans.predict(pca_df)
    # Create a DataFrame
    cluster_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])
    cluster_df['cluster'] = clusters
      # Save to CSV file
    cluster_df.to_csv('./data/cluster_results.csv', index=False)
    # 3D scatter plot
    fig = px.scatter_3d(
        cluster_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='cluster',
        color_continuous_scale='viridis',
        title='KMeans Clustering of Audio Features'
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='PC1'),
            yaxis=dict(title='PC2'),
            zaxis=dict(title='PC3')
        )
    )
    return fig

def preprocess_data(audio_waveforms):
    # Extract features for clustering
    # This part should be adapted to your specific requirements
    features = [get_mfcc(waveform, sr).numpy().flatten() for waveform, sr in audio_waveforms]
    
    # Create a DataFrame
    df = pd.DataFrame(features)

    # Standard Scaling
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # PCA
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    return pca_df

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

    # Visualize the selected feature
    if feature_option != "KMeans Clustering":
        for i, (waveform, sr) in enumerate(audio_waveforms):
            plot_feature(waveform, sr, feature_option, audio_files[i], labels[i])
    else:
        pca_df = preprocess_data(audio_waveforms)
        st.plotly_chart(plot_results(pca_df))

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

# Run this page
if __name__ == "__main__":
    demo_visualize_page()
