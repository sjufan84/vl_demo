""" Main page to record the audio clip to be processed."""
# Import required libraries
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from audio_recorder_streamlit import audio_recorder

# Establish the warning messages
RECORDING_NAME_WARNING = "**Please enter a name for the recording.**"
RECORDING_EXISTS_WARNING = "**A recording with that name already exists.\
Please enter a different name.**"
NO_AUDIO_WARNING = "**Please record audio before saving.**"
SUCCESS_MESSAGE = "**Recording saved successfully! You may now visualize\
the data by clicking the button below, or select another option.**"


# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'audio_files', 'record_audio_page', 'upload_audio_page'
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio'
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

def get_recordings():
    """ Get the uploaded files from the session state. """
    # Get the uploaded files
    st.success('**Click on the speaker to begin recording. Note: *Microphone\
            will turn red when recording and green when not.*\
    This is the initial step to grab a sample of the speakers voice.**')
    # Capture audio from the user's microphone
    audio_bytes = audio_recorder(recording_color="red", neutral_color="green", icon_size=100)
    # Check to see if audio was captured
    if audio_bytes is not None:
        st.audio(audio_bytes, format='audio/wav')
    st.markdown('---')
    st.success('**Once you are satisfied with your recording, enter\
                a name for it below and save the recording.**')
    # Allow the user to enter a name for the recording
    file_name = st.text_input("Recording Name:")
    save_recording_button = st.button("Save Recording", type='primary', use_container_width=True)

    if save_recording_button:
        if file_name == '':
            st.warning(RECORDING_NAME_WARNING)
        elif file_name in st.session_state.audio_files:
            st.warning(RECORDING_EXISTS_WARNING)
        elif audio_bytes is None:
            st.warning(NO_AUDIO_WARNING)
        else:
            with open(f"{file_name}.wav", 'wb') as f:
                f.write(audio_bytes)
            st.session_state.audio_files[file_name] = file_name
            st.success(SUCCESS_MESSAGE)
    st.markdown('---')
    
    # Create the buttons to switch pages
    visualize_data_button = st.button("Visualize Data", type = 'primary', use_container_width=True)
    upload_audio_button = st.button("Upload Audio", type='primary', use_container_width=True)
    record_another_button = st.button("Record Another", type='primary', use_container_width=True)

    # If the user clicks the upload audio button,
    # switch to the upload audio page
    if upload_audio_button:
        st.session_state.upload_audio_page = 'upload_audio'
        switch_page("Upload Audio")
    # If the user clicks the visualize data button,
    # switch to the visualize data page
    if visualize_data_button:
        switch_page("Visualize Data")
    # If the user clicks the record another button,
    # switch to the record audio page
    if record_another_button:
        st.session_state.record_audio_page = 'record_audio'
        st.experimental_rerun()
        
if st.session_state.record_audio_page == 'record_audio':
    get_recordings()

