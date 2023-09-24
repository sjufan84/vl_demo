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
        'chat_page', 'user_name', 'bplan_chat_page'
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio', {},'home', {}, {}, 'demo_visualize_home', "detailed_home",
        'nft_demo_home', 0, '', 0, {}, [], 'chat_home', '', 'bplan_home'
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

                
    st.markdown("""<div class="text-container">
  <h5>
    When a song imitating Drake and the Weeknd was released in April of this year
    that was <a href="https://www.nytimes.com/2023/04/19/arts/music/ai-drake-the-weeknd-fake.html"
    style="color:#5046B1">
    entirely generated by AI</a>, it revealed a reality that was
    once confined to <b>science fiction</b>. The music industry and
    Hollywood were thrown into disarray as <b>legal, ethical, and business
    dilemmas</b> surrounding artificial intelligence in the arts emerged.
  </h5>

  <br>

  <h5 style="color:#5046B1;"><i>Vocalockr responds to this challenge by empowering artists.</i></h5>
  <br>
  <h5>
      The reality is, it is relatively easy, with the right technical skills,
      to take less than <i>30 seconds</i> of an artist's vocals and create
      an AI model that can "swap" any other person's vocals for the artist's.</h5>
    <br>
    <h5 style="color:#5046B1"><i>We aim to protect artists by giving them the tools to not
    only protect themselves and their legacy, but also open up new and exciting revenue
    streams in this brave new world.</i></h5></div>""", unsafe_allow_html=True)
    st.text("")

    get_started_button = st.button("Get Started", type='primary', use_container_width=True)
    if get_started_button:
        switch_page("Demo Visualize")
    st.markdown('---')
    st.markdown("""
                **Curious about the business plan?  Click here to interact with the
                business plan assistant!**
                """)
    bplan_button = st.button("Business Plan", type='primary', use_container_width=True)
    if bplan_button:
        st.session_state.bplan_chat_page = 'bplan_home'
        switch_page("Business Plan")

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
