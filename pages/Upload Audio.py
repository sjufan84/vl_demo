""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
import streamlit as st
from streamlit_extras.switch_page_button import switch_page


def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    st.warning('**Upload vocal recordings below to visualize Melodic Voiceprint features.\
               These should be as high quality recordings as possible, vocals only, and ideally\
                shorter than 10 seconds.  Longer clips can be accomodated, but will take longer to\
                process.  Please ensure that the recordings are in .wav or .flac format.  If you would like\
                to experiment with recording audio, you may choose to do so below.**')
    uploaded_files = st.file_uploader("Upload recordings", type=["wav", "flac"], accept_multiple_files=True)
    if uploaded_files:
        upload_files_button = st.button("Upload File(s)", 
                                        type='primary', use_container_width=True)
        if upload_files_button:
            # Append each uploaded file to the list
            for uploaded_file in uploaded_files:
                # Save the file to the session state
                with open(uploaded_file.name, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.audio_files[uploaded_file.name] = uploaded_file
            # Display a success message
            st.success('**File(s) uploaded successfully!  You may now visualize the data\
                       by clicking the button below**')
                

    st.markdown('---')
    # Display buttons depending on whether or not the files were uploaded
    record_audio_button = st.button("Record Audio", type='primary', use_container_width=True)
    if record_audio_button:
        switch_page("Record Audio")
    visualize_data_button = st.button("Visualize Data", type='primary', use_container_width=True)
    if visualize_data_button:
        # Check to make sure that the user has uploaded at least one file
        if st.session_state.audio_files:
            st.session_state.visual_page = "has_audio"
            switch_page("Visualize Data")
        else:
            st.warning('**Please upload at least one audio file before proceeding.**')
        

    

if st.session_state.upload_audio_page == 'upload_audio':
    upload_audio()