""" Here we will demonstrate the voiceprint functionality of the app
by using a 3d plot of Joel and Jenny's voiceprints. """
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Vocal NFT Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="auto",
)

def voiceprint_visual():
    """ Voiceprint Homepage """
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">So Now What?</h4>
        <br>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Now that we have the artist's voiceprint secured,
        what else can we do with it?  Are we simply relegated to a world of playing
        defense against AI deepfakes?  Absolutely not!  While scary and disorienting,
        the advent of this technology actually opens up new and exciting ways that the artist,
        if they so choose, can create exciting new interactive experiences and opporunities for
        revenue generation.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">One such opportunity we are particularly excited about is "Co-writer",
        a revolutionary new path for artist engagement.  By leveraging multi-modal AI models
        that are trained <i>with the artist's input</i>, we aim to emulate a real-life co-writing session.
        <br>
        <br>
        We want to work <i>with</i> artists to select which data is used when creating their unique models. We are taking the opposite approach
        to companies whose models are a black box.  In fact, UMG is currently 
        <a href="https://www.billboard.com/pro/universal-music-sues-ai-company-using-songs-train-models/">suing Anthropic</a>,\
        an AI company with products similar to ChatGPT, for using copyrighted data in their training runs.  We choose to collaborate,
        rather than extract, to ensure that the artist is comfortable with the process <i>and</i> that those involved
        in the creation of the works used are compensated fairly.  If a song is used for training, we want to make sure
        royalties are distributed in a fair and ethical way.</h3>
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
                -o-animation: fadeIn ease 3s; -ms-animation:
                fadeIn ease 3s;">
                </div>""", unsafe_allow_html=True)
    
    st.text("")
   
    co_writer_button = st.button("Try Out Co-writer", type = 'primary', use_container_width=True)
    if co_writer_button:
        switch_page("Co-writer")

if __name__ == "__main__":
    voiceprint_visual()
      
    # Display the "Business Chat" sidebar
    from utils.bplan_utils import chat_state_sidebar
    chat_state_sidebar()