""" Home page for the application. Display the mechanism
to allow the user to upload or record a sample of their voice.  The
sample will then be used to generate the embeddings that will be used
to create the NFT."""

# Initial Imports
import base64
from streamlit_extras.switch_page_button import switch_page
import streamlit as st

st.set_page_config(
    page_title="Vocal NFT Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="auto",
)


# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'audio_files', 'record_audio_page', 'upload_audio_page',
        'embeddings', 'home_page', 'av_feature_dict', 'lc_feature_dict',
        'demo_visualize_page', "detailed_vocal_features_page",
        'nft_demo_page', 'token_id', 'contract_address',
        'latest_nft', 'nft_metadata', 'chat_history' 'chat_page', 'user_name', 'openai_model',
        'messages'
    ]
    default_values = [
        {}, 'record_audio', 'upload_audio', {}, 'home', {}, {},
        'demo_visualize_home', "detailed_home", 'nft_demo_home', 0,
        '', 0, {}, [], 'chat_home', '', 'bplan_home', "gpt-4-0613", []
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

if "secure_page" not in st.session_state:
    st.session_state["secure_page"] = "secure_home"
if "chat_state" not in st.session_state:
    st.session_state["chat_state"] = "off"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0613"

# Initialize the session variables
init_session_variables()

# Function to convert image to base64
def img_to_base64(img_path):
    """Convert an image to base64"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use the function
IMG_PATH = "./resources/artist_vault.png"  # Replace with your image's path
base64_string = img_to_base64(IMG_PATH)

def home():
    """ Home page for the application. Display the mechanism """
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img id='logo' src='data:image/png;base64,{base64_string}' style='height:80px; margin: 0;
        margin-right: 45px; animation: fadeIn ease 3s; -webkit-animation: fadeIn ease 3s; -moz-animation:
        fadeIn ease 3s; -o-animation: fadeIn ease 3s; -ms-animation: fadeIn ease 3s;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Empowering Artists in the Age of AI</h4>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                -o-animation: fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">
    <br>
    <h5>
    “When <a href="https://www.nytimes.com/2023/04/19/arts/music/ai-drake-the-weeknd-fake.html"
    style="color:#3D82FF">an AI deepfake</a> song imitating Drake and the Weeknd was released
    in April of last year, it revealed a reality that was once confined to <b>science fiction</b>.
    The music industry was thrown into disarray as <b>legal, ethical, and business
    dilemmas</b> surrounding artificial intelligence in the arts emerged.
    </h5>
    <h5 style="color:#3D82FF;"><i>First Rule responds to this challenge by empowering artists.</i></h5>
    <h5>
      The reality is, it is relatively easy, with the right technical skills,
      to take less than <i>30 seconds</i> of an artist's vocals and create
      an AI model that can "swap" any other person's vocals for the artist's.</h5>
    <h5 style="color:#3D82FF"><i>We aim to protect artists by giving them the tools to not
    only protect themselves and their legacy, but also open up new and exciting revenue
    streams in this brave new world.</i></h5></div>""", unsafe_allow_html=True)
    st.text("")

    get_started_button = st.button("Get Started", type='primary', use_container_width=True)
    if get_started_button:
        switch_page("Voice Swaps")

    st.text("")

    # Display the "Business Chat" sidebar
    from utils.bplan_utils import chat_state_sidebar
    chat_state_sidebar()

home()
