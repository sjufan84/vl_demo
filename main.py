""" Home page for the application. Display the mechanism
to allow the user to upload or record a sample of their voice.  The
sample will then be used to generate the embeddings that will be used
to create the NFT."""

# Initial Imports
from audio_recorder_streamlit import audio_recorder
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from PIL import Image

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'audio_files', 'record_audio_page', 'upload_audio_page', 'librosa_waveforms'
    ]
    default_values = [
        None, 'record_audio', 'upload_audio', {}
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    

def audio_home():
    """ Capture audio from the user's microphone or uploaded that will then
    need to be converted in .wav files if not already in that format. """
    # Create two columns to display the recorder and the upload mechanism
    st.markdown('''<div style="text-align: center;">
    <h2 class="title">Voice Locker Demo v1</h2>
                </div>''', unsafe_allow_html=True)
                
    st.markdown('---')
    st.markdown('''<div style="text-align: center;">
                <h3 class="title">Choose an option below to get started.</h3>
                </div>''', unsafe_allow_html=True)
    
    st.text('')
    col1, col2 = st.columns(2, gap="medium")

    # Display the recorder in the first column
    
    with col1:
         # Use PIL to read in the image
        mic_image =  Image.open('./resources/studio_mic1.png')
        st.image(mic_image, use_column_width=True)
        choose_recording_button = st.button("Record Audio", type='primary', use_container_width=True)
        if choose_recording_button:
            st.session_state.get_audio_page = 'record_audio'
            st.experimental_rerun()

       
            
    # Display the up
    # load mechanism in the second column
    with col2:
        # Create the upload mechanism
        # Use PIL to read in the image
        upload_image = Image.open('./resources/record_upload1.png')
        st.image(upload_image, use_column_width=True)
        upload_files_button = st.button("Upload Files", type='primary', use_container_width=True)
        if upload_files_button:
            st.session_state.get_audio_page = 'upload_audio'
            st.experimental_rerun()
            

            


if st.session_state.get_audio_page == "audio_home":
    audio_home()
elif st.session_state.get_audio_page == "record_audio":
    get_recordings()
elif st.session_state.get_audio_page == "upload_audio":
    upload_audio()
    