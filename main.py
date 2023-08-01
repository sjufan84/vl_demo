""" Home page for the application. Display the mechanism
to allow the user to upload or record a sample of their voice.  The
sample will then be used to generate the embeddings that will be used
to create the NFT."""

# Initial Imports
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from PIL import Image

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'audio_files', 'record_audio_page', 'upload_audio_page', 'waveform_data', 'features', 'tsne_results', 'embeddings', 'visual_page', 'home_page'
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio', {}, {}, {}, {}, 'visual_home', 'home'
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    

def home():
    """ Landing page for the application. """
    # Create two columns to display the recorder and the upload mechanism
    st.markdown("""<div style="text-align: center"><h1>Voice Lockr Demo v1</h1></div>""", unsafe_allow_html=True)
                
    st.markdown('---')
    st.markdown("""**Our voice print technology transforms the realm of audio and voice content. We leverage advanced algorithms to create unique voice prints - think of it as the 'fingerprint' of one's voice. When copyright disputes arise, our technology can compare voice prints to detect potential infringements, providing an unprecedented level of accuracy in voice-related disputes.**""")

    st.markdown("""**Imagine creating personalized voice assistants or virtual avatars using a unique voice - that's the power of our voice print technology in AI content generation. We can train AI models on specific voice prints, allowing the generation of new content that retains unique voice characteristics. In addition, our technology ensures that artists' voices are not used without their consent, paving the way for a new era of ethical, personalized content creation.**""")
    st.markdown('---')
    get_started_button = st.button("Get Started", type='primary', use_container_width=True)
    if get_started_button:
        st.session_state.home_page = 'audio_home'
        st.experimental_rerun()

def audio_home():
    """ Capture audio from the user's microphone or uploaded that will then
    need to be converted in .wav files if not already in that format. """    
    st.text('')
    col1, col2 = st.columns(2, gap="medium")

    # Display the recorder in the first column
    
    with col1:
         # Use PIL to read in the image
        mic_image =  Image.open('./resources/studio_mic1.png')
        st.image(mic_image, use_column_width=True)
        choose_recording_button = st.button("Record Audio", type='primary', use_container_width=True)
        if choose_recording_button:
            st.session_state.record_audio_page = 'record_audio'
            switch_page("record_audio")
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
            st.session_state.upload_audio_page = 'upload_audio'
            switch_page("upload_audio")
            st.experimental_rerun()

if st.session_state.home_page == 'home':
    home()
elif st.session_state.home_page == 'audio_home':
    audio_home()