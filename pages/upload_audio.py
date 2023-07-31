""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
from PIL import Image
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import get_waveform_data, extract_features
import torchaudio
import os

if 'progress' not in st.session_state:
    st.session_state.progress = 0


def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    # Use PIL to read in the image
    upload_image = Image.open('./resources/record_upload1.png')
    st.image(upload_image)

    st.warning('**Click on the button below to upload a file from your\
            local machine.  This is the second step / option to grab a sample of\
            the speakers voice.  This sample should be in .wav format**')
    uploaded_files = st.file_uploader("Upload a file", type="wav", accept_multiple_files=True)
    
    if uploaded_files:
        upload_files_button = st.button("Upload File", type='primary', use_container_width=True)
        if upload_files_button:
            with st.spinner('Processing file...'):
                for uploaded_file in uploaded_files:
                    # Save the file to the session state
                    with open(uploaded_file.name, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    # Get the file
                    file = uploaded_file.name
                    file_name = os.path.basename(file) # Assuming file_name is the base name of the file
                    st.session_state.audio_files[file_name] = file
                    # Update the waveform data
                    st.session_state.waveform_data[file_name] = get_waveform_data(file)
                    # Extract the features from the audio file
                    features = extract_features(file)
                    st.session_state.features_data[file_name] = features
                    #st.write(st.session_state.features_data[file_name])
                    # Once all of the files have been processed, display a success message
                st.success("File(s) uploaded successfully!")

    # Display the buttons
    upload_more_button = st.button("Upload more files", type='primary', use_container_width=True)
    if upload_more_button:
        # If the user wants to upload more files, reset the progress bar
        st.experimental_rerun()
    record_audio_button = st.button("Record Audio", type='primary', use_container_width=True)
    if record_audio_button:
        # If the user wants to record audio, switch to the record_audio page
        switch_page("record_audio")
        st.experimental_rerun()
    visualize_data_button1 = st.button("Visualize Data", type='primary', use_container_width=True)
    if visualize_data_button1:
        # If the user wants to visualize data, switch to the visualize_data page
        switch_page("visualize_data")

if st.session_state.upload_audio_page == 'upload_audio':
    upload_audio()