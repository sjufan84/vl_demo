""" Home page for the application. Display the mechanism
to allow the user to upload or record a sample of their voice.  The
sample will then be used to generate the embeddings that will be used
to create the NFT."""

# Initial Imports
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Vocal NFT Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'audio_files', 'record_audio_page', 'upload_audio_page','embeddings', 'home_page',
        'av_feature_dict', 'lc_feature_dict', 'demo_visualize_page', "detailed_vocal_features_page",
        'nft_demo_page', 'token_id', 'contract_address', 'latest_nft', 'nft_metadata', 'chat_history'
        'chat_page', 'user_name'
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio', {},'home', {}, {}, 'demo_visualize_home', "detailed_home",
        'nft_demo_home', 0, '', 0, {}, [], 'chat_home', ''
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    


def home():
    """ Landing page for the application. """
    # Display the description
    st.markdown("""<div>
                <h4 style="font-family: 'Montserrat', sans-serif; color: #2F3A56; font-size: 40px; font-weight: 550;">
                Vocalockr: Empowering Artists in the Age of AI</h4>
                </div>""", unsafe_allow_html=True)
    st.markdown("---")

                
    st.markdown("""<div>
                When a song, <i>imitating Drake and the Weeknd through AI</i>, 
                shocked social media this April, it revealed a reality that was
                once confined to <b>science fiction</b>. The music industry and
                Hollywood were thrown into disarray as <b>legal, ethical, and business
                dilemmas</b> surrounding artificial intelligence in the arts emerged.<br><br>
                <h5>Vocalockr responds to this challenge by empowering artists.</h5>
                Utilizing the technology behind <i>deepfake creations</i>, Vocalockr enables
                artists to generate <b>Melodic Voiceprints</b> and store them securely on the
                <b>blockchain</b>. By doing so, we place control over voice licensing firmly
                in the hands of the creators themselves in this rapidly evolving world
                of content generation. <br><br>
                <h5>In this demo, we will guide you through the MV
                capture process and explore the wide-ranging applications of this
                groundbreaking technology.</h5> Join us as we showcase how <b>Vocalockr</b>
                restores autonomy to artists, navigating them through this <i>brave new era</i>.
                </div>""", unsafe_allow_html=True)
    st.markdown('---')
    get_started_button = st.button("Get Started", type='primary', use_container_width=True)
    if get_started_button:
        switch_page("Demo Visualize")

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
            switch_page("Record Audio")
       
            
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
            switch_page("Upload Audio")
            st.experimental_rerun()

home()