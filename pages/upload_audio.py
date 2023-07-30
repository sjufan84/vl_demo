""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
from PIL import Image
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import get_waveform_data


def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    # Use PIL to read in the image
    upload_image = Image.open('./resources/record_upload1.png')
    st.image(upload_image)

    st.warning('**Click on the button below to upload a file from your\
            local machine.  This is the second step / option to grab a sample of\
            the speakers voice.  This sample should be in .wav format**')
    uploaded_file = st.file_uploader("Upload a file", type="wav")
    file_name  = st.text_input("Enter a name for the recording:")
    if uploaded_file:
        upload_files_button = st.button("Upload File")
        if upload_files_button:
            # Save the file to the session state
            st.session_state.audio_files[file_name] = uploaded_file
            # Create the waveform data
            # Create  a progress bar to display the and processing status
            # First for saving the file to the session state
            # Then for converting the file to waveform data
            # with the librosa library
            progress_bar = st.progress(0)
            progress_text = st.empty()
            # Update the progress bar and text
            progress_bar.progress(50)   
            progress_text.text("Saving file...")
            
