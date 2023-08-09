""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
import os
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import get_waveform_data, extract_norm_features, get_embeddings
from PIL import Image

if 'progress' not in st.session_state:
    st.session_state.progress = 0


def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    st.warning('**Upload vocal recordings below to visualize Melodic Voiceprint features.\
               These should be as high quality recordings as possible, vocals only, and ideally\
                shorter than 10 seconds.  Longer clips can be accomodated, but will take longer to\
                process.  Please ensure that the recordings are in .wav format.  If you would like\
                to experiment with recording audio, you may choose to do so below.**')
    uploaded_files = st.file_uploader("Upload recordings", type="wav", accept_multiple_files=True)
    
    #if uploaded_files:
    upload_files_button = st.button("Upload File(s) and visualize data", type='primary', use_container_width=True)
    if upload_files_button:
        with st.spinner('Processing recordings...'):
                for uploaded_file in uploaded_files:
                    # Save the file to the session state
                    with open(uploaded_file.name, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                     # Get the file
                    file = uploaded_file.name
                    # Save the file to the session state
                    st.session_state.audio_files["user"] = file
                    # Update the waveform data
                    st.session_state.waveform_data["user"] = get_waveform_data(file)
                    # Extract the embeddings from the audio file
                    st.session_state.embeddings["user"] = get_embeddings(file)
                    # Extract the features from the audio file
                    st.session_state.features["user"] = extract_norm_features(file)
                # Once all of the files have been processed, display a success message
                st.success("File(s) uploaded successfully!")
                switch_page("visualize_data")

    record_audio_button = st.button("Record Audio", type='primary', use_container_width=True)
    if record_audio_button:
        # If the user wants to record audio, switch to the record_audio page
        switch_page("record_audio")
    #visualize_data_button1 = st.button("Visualize Data", type='primary', use_container_width=True)
    #if visualize_data_button1:
    #    st.session_state.visual_page='visual_home'
        # If the user wants to visualize data, switch to the visualize_data page
    #    switch_page("visualize_data")

if st.session_state.upload_audio_page == 'upload_audio':
    upload_audio()