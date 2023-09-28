""" Demo page for visualizing audio features """ 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import librosa
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import extract_features, read_audio

st.set_page_config(page_title="Voice Lockr Demo", page_icon=":microphone:",
                initial_sidebar_state="collapsed", layout="wide")


def get_3d_chart_ghost():
    """ Get the 3D chart for Fast Car """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/jenny_ghost.wav')
    lc_signal = read_audio('./audio_samples/ella_ghost.wav')
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
    lc_labels = [f"EH - Segment {i+1}" for i in range(len(lc_segments))]
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
        st.markdown("**Ella Henderson**")
        st.audio(lc_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Joel**")
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Jenny**")
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        options=['Jenny', 'EH', 'Joel'],
        default=['Jenny', 'EH', 'Joel'],
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
        title='3D Representation of Vocal Features "Ghost" -- EH, Joel, Jenny',
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
        marker_size=8,            # Font size of the text labels
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
    if len(jenny_signal) < 2048 or len(lc_signal) < 2048 or len(joel_signal) < 2048:
        st.warning("One of the audio signals is too short for FFT. Skipping...")
        return

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
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 15px; font-weight: 550;">
                    LC Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 15px; font-weight: 550;">
                    Joel Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 15px; font-weight: 550;">
                    Jenny Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(lc_bytes, format='audio/wav', start_time=0, sample_rate=16000)
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
                <div class="text-container">
                <h4 style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
                How it Works</h4>
                </div>""", unsafe_allow_html=True)
                

    st.markdown("""**Now, let's see what happens when we swap out the voices of Joel and Jenny for another artist's.
    There are two clips we have used for this demo.  The first is Luke Combs's rendition of "Fast Car", 
    and the other is Ella Henderson's "Ghost".  We can swap out male for female voices
    by utilizing the same methodology of other voice clones and simply adjust the pitch to create the best
    version.  This proves that this technology is here today, evolving, and constantly getting more 
    powerful.**""")
    st.text("")
    st.markdown("""**The 3d Chart below illustrates each artist's vocal print at different points
                in the song.  The segments are color coded by artist, and the labels on the chart indicate the
                artist and segment number.  In order to visualize the differences in the vocal prints, we have
                used a technique called Principal Component Analysis (PCA) to reduce the dimensionality of the
                data.  The actual number of features is much higher, but using PCA allows us to capture most of the
                variance in the data with just 3 features.  Even though they are singing the same song, there are
                distinct characteristics for each voice, and those differences are what makes each voiceprint unique.  
                For a better visual on different clusters, the chart can be rotated and panned.**
                """)
    # Create a selectbox to allow the user to select which song to visualize
    song = st.selectbox("Select a Song to Visualize:", ["Fast Car", "Ghost"])
    if song == "Fast Car":
        get_3d_chart_fcar()
    else:
        get_3d_chart_ghost()

    # Create a button to switch to the next page
    continue_to_mv_button = st.button("Continue", type="primary", use_container_width=True)
    if continue_to_mv_button:
        st.session_state.secure_page = "secure_home"
        switch_page("Secure")
        st.experimental_rerun() 

if __name__ == "__main__":
    voice_swap_home()