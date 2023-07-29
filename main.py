""" Home page for the application. Display the mechanism
to allow the user to upload or record a sample of their voice.  The
sample will then be used to generate the embeddings that will be used
to create the NFT."""

# Initial Imports
from audio_recorder_streamlit import audio_recorder
import streamlit as st
from PIL import Image

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'samples', 'get_audio_page'
    ]
    default_values = [
        {}, "audio_home"
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
            
def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    # Use PIL to read in the image
    upload_image = Image.open('./resources/record_upload1.png')
    st.image(upload_image)

    st.warning('**Click on the button below to upload a file from your\
            local machine.  This is the second step / option to grab a sample of\
            the speakers voice.  These can be .wav files for now.**')
    uploaded_file = st.file_uploader("Upload a file", type=["wav"], accept_multiple_files=True)
    if uploaded_file:
        upload_files_button = st.button("Upload Files")
        if upload_files_button:
            # Display the uploaded files
            st.success("**Files uploaded!  The list of your\
                    current samples is below:**")
            # Display the key value pairs of the session state
            for uploaded_file in uploaded_file:
                st.markdown(f"**{uploaded_file.name}**")
                st.audio(uploaded_file, format='audio/wav')

def get_recordings():
    """ Get the uploaded files from the session state. """
    # Get the uploaded files
    mic_image =  Image.open('./resources/studio_mic1.png')
    st.image(mic_image, use_column_width=True)
    st.warning('**Click on the speaker to begin recording. Note: *Microphone\
            will turn red when recording and green when not.*\
    This is the initial step to grab a sample of the speakers voice.**')
    # Capture audio from the user's microphone
    audio_bytes = audio_recorder(recording_color="red", neutral_color="green", icon_size=100)
    # Check to see if audio was captured
    if audio_bytes is not None:
        st.audio(audio_bytes, format='audio/wav')
    st.markdown('---')
    st.markdown('**Once you are satisfied with your recording, enter a\
            name for the recording and click the\
                "Save Recording" button to save the recording.  You can then\
                add other recordings from the microphone or upload files\
                to make the sample more robust.**')
    # Allow the user to enter a name for the recording
    recording_name = st.text_input("Recording Name:")
    # Check to make sure a name and audio was entered,
    # then display a button to save the recording
    if recording_name != "" and audio_bytes is not None:
        save_recording_button = st.button("Save Recording",
        type = 'primary', use_container_width=True)
        if save_recording_button:
            # Add the recording to the session state using
            # the name of the recording as the key
            st.session_state.samples[recording_name] = audio_bytes
            # Display the current samples
            st.success("**Recording saved!  The list of your\
                    current samples is below:**")
            # Display the key value pairs of the session state
            for key, value in st.session_state.samples.items():
                st.markdown(f"**{key}**")
                st.audio(value, format='audio/wav')
    else:
        st.markdown('---')
        st.warning("**Please ensure that you have entered a name for the\
                recording and the recording has been captured.**")


if st.session_state.get_audio_page == "audio_home":
    audio_home()
elif st.session_state.get_audio_page == "record_audio":
    get_recordings()
elif st.session_state.get_audio_page == "upload_audio":
    upload_audio()
    