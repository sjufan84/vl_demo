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
        'audio_files', 'record_audio_page', 'upload_audio_page','embeddings', 'home_page',
        'av_feature_dict', 'lc_feature_dict', 'demo_visualize_page', "detailed_vocal_features_page"
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio', {},'home', {}, {}, 'demo_visualize_home', "detailed_home"
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
    st.markdown("""<div><p><b>In April of this year, </b>when an AI generated song imitating Drake and
                the Weekend took <a href="https://www.nytimes.com/2023/04/19/arts/music/ai-drake-the-weeknd-fake.html">social media by storm</a>,
                it set off alarm bells across the music industry.  What seemed like the stuff of science fiction only a short time ago was now
                all too real.  Shortly thereafter, Hollywood confronted its own battles over AI.  Actors and writers
                decided to go on strike, in part because of the possibility that their likenesses, which in some cases had been signed away
                in contracts written before the dawn of this new age, were suddenly at risk of being used to create content
                that they would not even need to be present for.  This, of course, raises all sorts of legal, ethical, and business
                questions about the future of artificial intelligence in the arts.  What, if any, recourse do artists have to protect themselves
                in this environment?<br><br>
                Thankfully, the same technology that is used to produce these deepfake songs can also be used to help artists fight back
                against them.  This is what Vocalockr aims to do.  By helping artists generate the same Melodic Voiceprints that fuel these
                AI generated voice clones and storing it on the blockchain, we put the power back in the hands of the artists to decide when, where,
                and how to license their voice in this brave new world of content generation.  This demo will walk through the stages of the MV capture
                process as well as the myriad use cases that this technology represents.
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

if st.session_state.home_page == 'home':
    home()
elif st.session_state.home_page == 'audio_home':
    audio_home()