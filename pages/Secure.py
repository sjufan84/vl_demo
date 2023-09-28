""" Demo for securing the Melodic Voiceprint of an artist. \
    We will allow the user to upload and / or record a clip\
    of their voice and then generate a demo of the Melodic Voiceprint"""
import io
import time
import soundfile as sf
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import extract_features

# Set the page configuration
st.set_page_config(
    page_title="Melodic Voiceprint Demo",
    page_icon="🎤",
    initial_sidebar_state="collapsed",
)


if "audio_bytes_list" not in st.session_state:
    st.session_state["audio_bytes_list"] = []
if "fig" not in st.session_state:
    st.session_state["fig"] = None
if "secure_page" not in st.session_state:
    st.session_state["secure_page"] = "secure_home"

def read_audio(audio_bytes: bytes) -> np.ndarray:
    """ Read the audio bytes into a NumPy array """
    with io.BytesIO(audio_bytes) as f:
        audio_array = sf.read(f)
    return audio_array[0] # Return the audio array

def record_audio():
    """ Capture audio from the user's microphone. """
    st.title("Record Audio")
    audio_bytes = audio_recorder(text="Click to record")
    st.audio(audio_bytes)

def generate_3d_plot():
    """ Generate a 3D plot of the Melodic Voiceprint for the artist """
    # If there are 2 audio clips, concatenate them
    total_signal = None
    if len(st.session_state.audio_bytes_list) == 2:
        # Convert the tuples to lists
        st.session_state.audio_bytes_list[0] = list(st.session_state.audio_bytes_list[0])
        st.session_state.audio_bytes_list[1] = list(st.session_state.audio_bytes_list[1])
        # Flatten the audio clips if necessary 
        for i in range(2):
            if st.session_state.audio_bytes_list[i][0].ndim > 1:
                st.session_state.audio_bytes_list[i][0] = np.mean(st.session_state.audio_bytes_list[i][0], axis=1)
        total_signal = np.concatenate([st.session_state.audio_bytes_list[0][0],
                                    st.session_state.audio_bytes_list[1][0]])
    else:
        total_signal = st.session_state.audio_bytes_list[0][0]

    # Extract the features from the audio clip
    features = extract_features(total_signal)
 
    # Transpose to have features as columns
    signal_features = features.T

    # Flatten the features to 2d if necessary
    if signal_features.ndim > 2:
        signal_features = signal_features.reshape(signal_features.shape[0], -1)

    # Create a DataFrame from the features
    df = pd.DataFrame(signal_features)
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])
    
    # Standardize the data before applying PCA
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])

    # Split the DataFrame into segments
    mv_segments = np.array_split(plot_df, 3)


    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(mv_segments))]

    # Add segment names and numbers to the DataFrame
    for i, segment in enumerate(mv_segments):
        segment['segment_name'] = segment_numbers[i]
        segment['segment_number'] = i+1
    
    # Plot the 3D plot
    fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3', color=plot_df.index, hover_name=plot_df.index,
                        color_continuous_scale=px.colors.sequential.Agsunset, opacity=0.5, width=450, height=450)

    # Update the layout
    fig.update_layout(title="Your Melodic Voiceprint", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))

    return fig

def secure_home():
    """ The main app function for the Secure page """
    st.markdown("""##### Upload an audio clip or record your voice below\
                to generate a sample of your Melodic Voiceprint.  Obviously\
                the actual training process is much more involved, but this\
                will still provide a sense of how the process works.
                """)
    st.text("")
    st.text("")
    # Create two columns, one for uploading, and one for recording
    col1, col2 = st.columns(2, gap="large")
    with col1:
        mic_image = Image.open('./resources/studio_mic1_small.png')
        st.image(mic_image, use_column_width=True)
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        
        col3, col4, col5 = st.columns([0.5, 1.5, 1])
        with col3:
            st.text("")
        with col4:
            st.markdown("Click to record: :green[Stopped] / \
                        :red[Recording]")
        with col5:
            recorded_audio_bytes = audio_recorder(text="", icon_name="record-vinyl",
            sample_rate=16000, neutral_color = "green", icon_size="3x")
        st.text("")
        st.text("")
        st.text("")
        if recorded_audio_bytes:
            st.markdown("""
                        <p style="color:#EDC480; font-size: 23px; text-align: center;">
                        Recorded audio clip:
                        </p>
                        """, unsafe_allow_html=True)
            st.audio(recorded_audio_bytes)
            recorded_audio = sf.read(io.BytesIO(recorded_audio_bytes))
            # Using soundfile to read the audio file into a NumPy array
            st.session_state.audio_bytes_list.append(recorded_audio)

    with col2:
        upload_image = Image.open('./resources/record_upload1_small.png')
        st.image(upload_image, use_column_width=True)
        st.text("")
        uploaded_file = st.file_uploader("Upload an Audio File", type=['wav'],
                                        key='upload_audio')
        if uploaded_file:
            with io.BytesIO(uploaded_file.getbuffer()) as f:
                # Using soundfile to read the audio file into a NumPy array
                audio = sf.read(f)
                if audio:
                    st.audio(audio[0], sample_rate=audio[1])
                st.session_state.audio_bytes_list.append(audio)

    st.markdown("""
                **Once you have uploaded and / or recorded your audio,\
                click below to generate your Melodic Voiceprint.**
                """)
    
    generate_mv_button = st.button("Generate Melodic Voiceprint",
    type='primary', use_container_width=True)
    if generate_mv_button:
        # Generate the 3D plot
        st.session_state.fig = generate_3d_plot()
        # Switch to the plot page
        st.session_state.secure_page = "secure_plot"
        st.experimental_rerun()

def secure_plot():
    """ Display the user generated Melodic Voiceprint """
    st.markdown("""
    <div style="font-size: 15px">
    <h5>
    Congratulations!  You have successfully\
    generated your Melodic Voiceprint and it is ready\
    to be secured.  The 3d plot below is a basic\
    representation of your unique vocal signature,\
    distilled down to three principal components for visualization\
    purposes.  Of course, the actual voiceprint that is generated\
    from longer training runs is much more robust.</h5>
    <h5>
    When you are done reviewing the plot, secure your MV and proceed\
    to the next step! 
    </h5>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Display the 3D plot
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    secure_button = st.button("Secure Your Voiceprint", type='primary',
    use_container_width=True)
    if secure_button:
        # Count down from 3 to 1 and then display the secure message
        for i in range(3, 0, -1):
            time.sleep(1.5)
            st.markdown(f"""
            <div style="font-size: 30px;">
            <h5 style="text-align: center">
            {i}
            </h5>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
        switch_page("Voiceprint Demo")
    create_new_mv_button = st.button("Create a new Melodic Voiceprint",
    type='primary', use_container_width=True)
    if create_new_mv_button:
        st.session_state.audio_bytes_list = []
        st.session_state.secure_page = "secure_home"
        st.experimental_rerun()

            

            
if st.session_state.secure_page == "secure_home":
    secure_home()
elif st.session_state.secure_page == "secure_plot":
    secure_plot()
