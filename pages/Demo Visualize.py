""" Demo page for visualizing audio features """
import tempfile
import base64   
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Voice Lockr Demo", page_icon=":microphone:", initial_sidebar_state="collapsed", layout="wide")

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'demo_visualize_page'
    ]
    default_values = [
        'demo_visualize_home'
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    

def audio_player_component():
    """ Audio player component """
    # specify directory and initialize st_audiorec object functionality
    audio_player = components.declare_component("audio_player", url="http://localhost:3001")

    return audio_player

def read_audio(file_path):
    """ Read an audio file and return the signal """
    signal, _ = librosa.load(file_path, sr=16000)
    return signal

# Create temporary audio files for the segments
def create_audio_files(segments, sr=16000):
    """ Create temporary audio files for the segments """
    files = []
    for segment in segments:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, segment, samplerate=sr)
        files.append(tmp_file)
    return files

# Create a data URL for an audio file
def audio_file_to_data_url(file):
    """ Create a data URL for an audio file """
    audio_encoded = base64.b64encode(file.read())
    return "data:audio/wav;base64," + audio_encoded.decode()

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

def create_audio_urls(segments):
    """ Create data URLs for the audio segments """
    urls = []
    for segment in segments:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        librosa.output.write_wav(tmp_file.name, segment, sr=16000)
        with open(tmp_file.name, "rb") as file:
            audio_encoded = base64.b64encode(file.read())
        urls.append("data:audio/wav;base64," + audio_encoded.decode())
    return urls

def demo_visualize():
    """ Demo page for visualizing audio features via Kmeans clustering """
    st.markdown("""
    ### Melodic Voiceprint: A Harmony of Science, Art, and Security
                
    **The logical first question to ask is:**  How do these deepfake audio clips work,\
    and what can artists do to protect themselves?  The answer lies in securing the\
    Melodic Voiceprint of the artist that is being used to train the models\
    that make these deepfakes possible.

    **The 3D chart below** visualizes the unique features that compose the voice of two different artists.
    Each point represents a segment of a song, and the position of the points reflects various
    characteristics of the voice such as pitch, rhythm, and timbre.
    The segments are grouped by color, highlighting similarities and differences between the artists.  By\
    securing their MV with Vocalockr on the blockchain, artists can ensure that their voiceprint is protected.
                
    **It's worth noting that** these two clips illustrate two primary use cases for Vocalockr.  The rendition\
                of "Happy Birthday" illustrates the possibilities for personalized content generation for fans,\
                just one of many implementations that could produce previously unimaginable revenue streams for artists.\
                Conversely, however, this shows just how easy it would be for anyone to create a personal and PR\
                nightmare for the artist.  Anyone with the right set of skills could create a deepfake of an artist\
                singing virtually anything, without any immediately verifiable proof that they were\
                not licensed to do so.
    """)

    # Read and preprocess audio signals
    #avicii_signal = read_audio('./audio_samples/avicii1.wav')
    combs_signal = read_audio('./audio_samples/happy_bday.wav')
    jeremiah_signal = read_audio('./audio_samples/jeremiah2.wav')
    min_length = min(len(combs_signal), len(jeremiah_signal))
    #avicii_signal = avicii_signal[:min_length]
    combs_signal = combs_signal[:min_length]
    jeremiah_signal = jeremiah_signal[:min_length]
    

    # Extract features
    #avicii_features = extract_features(avicii_signal)
    combs_features = extract_features(combs_signal)
    jeremiah_features = extract_features(jeremiah_signal)

    # Transpose to have features as columns
    #avicii_features = avicii_features.T
    combs_features = combs_features.T
    jeremiah_features = jeremiah_features.T

    # Create a DataFrame by concatenating the two feature sets
    #df = pd.concat([pd.DataFrame(avicii_features), pd.DataFrame(combs_features), pd.DataFrame(jeremiah_features)], axis=0)
    df = pd.concat([pd.DataFrame(combs_features), pd.DataFrame(jeremiah_features)], axis=0)
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

    # Keep every other row for each artist
    #cluster_df = cluster_df.iloc[::2, :]

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    #avicii_segments = np.array_split(avicii_signal, len(cluster_df)//3)
    combs_segments = np.array_split(combs_signal, len(cluster_df)//2)
    jeremiah_segments = np.array_split(jeremiah_signal, len(cluster_df)//2)
    #total_segments = avicii_segments + combs_segments + jeremiah_segments
    total_segments = combs_segments + jeremiah_segments

   # Create labels for the segments
    #avicii_labels = [f"Avicii - Segment {i+1}" for i in range(len(avicii_segments))]
    combs_labels = [f"User - Segment {i+1}" for i in range(len(combs_segments))]
    jeremiah_labels = [f"Jeremiah - Segment {i+1}" for i in range(len(jeremiah_segments))]
    #segment_labels = avicii_labels + combs_labels + jeremiah_labels
    segment_labels = combs_labels + jeremiah_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(combs_segments))] * 2

    # Add segment names and numbers to the DataFrame
    cluster_df['segment_name'] = segment_labels
    cluster_df['segment_number'] = segment_numbers


    
    # Create temporary audio files
    #avicii_files = create_audio_files(avicii_segments)
    combs_files = create_audio_files(combs_segments)
    jeremiah_files = create_audio_files(jeremiah_segments)

    # Create audio URLs
    #avicii_audio_urls = [audio_file_to_data_url(file) for file in avicii_files]
    combs_audio_urls = [audio_file_to_data_url(file) for file in combs_files]
    jeremiah_audio_urls = [audio_file_to_data_url(file) for file in jeremiah_files]
    #audio_urls = avicii_audio_urls + combs_audio_urls + jeremiah_audio_urls
    audio_urls = combs_audio_urls + jeremiah_audio_urls

    col1, col2 = st.columns([1.75, 1], gap='large')


    with col2:# Add a Streamlit multiselect widget to allow users to select artists
        st.text("")
        st.text("")
        st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa
       
        st.markdown("**Original Audio Clips:**")
        jeremiah_bytes = librosa.to_mono(jeremiah_signal)
        combs_bytes = librosa.to_mono(combs_signal)
        st.markdown("**User Happy Birthday**")
        st.audio(combs_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("**Jeremiah Harmon Happy Birthday**")
        st.audio(jeremiah_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        #options=['Avicii', 'Luke Combs', 'Jeremiah'],
        #default=['Avicii', 'Luke Combs', 'Jeremiah'],
        options=['User', 'Jeremiah'],
        default=['User', 'Jeremiah'],
        )
        # Filter the DataFrame based on selected artists
        filtered_cluster_df = cluster_df[cluster_df['segment_name'].str.contains('|'.join(selected_artists))]

        st.text("")
        st.text("")

        st.success("""**To drill down even further, select a segment to play the audio clip\
                 that corresponds to that segment.**""")
        # Create a selectbox to allow the users to choose the segment to play
        # We only want to show the unique segment names apart from the artist name
        # For example, we want to show "Segment 1" instead of "Avicii - Segment 1"
        unique_segment_names = list(set([segment.split(' - ')[1] for segment in cluster_df['segment_name']]))
        # List the unique segment names in ascending order of the segment number
        unique_segment_names.sort(key=lambda x: int(x.split(' ')[1]))
        segment_options = st.selectbox(
        "Select Segment:",
        options=unique_segment_names,
        )
        # Let the user choose which artist to play
        selected_artist = st.selectbox(
        "Select Artist:",
        #options=['Avicii', 'Luke Combs', 'Jeremiah'],
        options = ['User', 'Jeremiah'],
        )
      
        selected_segment = f"{selected_artist} - {segment_options}"
        # Convert the selected artist to the corresponding audio URL
        # Get the index of the selected label in the segment_labels list
        selected_index = segment_labels.index(selected_segment)

        # Convert the audio from bytes to a playable file using librosa
        audio_bytes = librosa.to_mono(total_segments[selected_index].T)
        st.audio(audio_bytes, format='audio/ogg', start_time=0, sample_rate=16000)



    with col1:# Plot using the filtered DataFrame
        fig = px.scatter_3d(
        filtered_cluster_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Vocal Features -- Luke Combs and Jeremiah Harmon',
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
            
    #st.markdown("""
    #            ***If you are interested in viewing even more granular details of the audio, you can
    #            click the button below.***
    #            """)
        # Create a button for showing the detailed audio features
    #detailed_features_button = st.button("Show Detailed Audio Features", type="primary", use_container_width=True)
    #if detailed_features_button:
    #    switch_page("Detailed Vocal Features")
    st.markdown("""---""")
    st.markdown("""
                 **So What Does This Mean for Music, Security, and the Future of the Industry?**

    1. **Understanding the Voice**: By analyzing these features, we can create a "Melodic Voiceprint,"
                a unique signature of an artist's voice. It's like a fingerprint for their voice,
                capturing the subtle nuances that make their voice distinctly theirs.

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

    st.text("")
    st.markdown("""
                **By securing the Melodic Voiceprint through NFTs**, or non-fungible tokens,
                Vocalockr ensures unique and protected ownership. An NFT represents a binding
                contract between an artist and an owner, whether a record label, streaming
                service, or fan. Without owning the NFT, usage of the artist's voice is
                unapproved. This method not only safeguards the artist's voice but also
                guarantees that it's used in line with their wishes, offering a powerful
                tool in the evolving digital landscape of music.
                """)
    st.text("")
    mint_nft_button = st.button("Mint an MV NFT", type="primary", use_container_width=True)
    if mint_nft_button:
        st.session_state.nft_demo_page = "nft_demo_home"
        switch_page("Generate NFT")


if st.session_state.demo_visualize_page == "demo_visualize_home":
    demo_visualize()
