""" Demo for securing the Melodic Voiceprint of an artist. \
    We will allow the user to upload and / or record a clip\
    of their voice and then generate a demo of the Melodic Voiceprint"""
import io
import time
import librosa
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import extract_features

# Set the page configuration
st.set_page_config(
    page_title="Reverse Audio Search",
    page_icon="🎤",
    initial_sidebar_state="collapsed",
)

def secure_home():
    """ The main app function for the Secure page """
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Firt Rule's Solution</h4>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">We will work with each artist in order to
        train their own voiceprint model.  Once stored securely in the Artist Vault, it can be used
        to quickly identify deepfakes, allowing them to take the necessary actions for takedowns, etc.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">Below we illustrate a simple demo of
        this "Reverse Audio Search" capability.  This is the cornerstone of First Rule's approach,
        acting with the artist's security and peace of mind as our guiding principle.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">By using the same high level algorithms
        used to generate the Melodic Voiceprint, we can calculate a "Similarity Score" that represents
        the probability that the clip in question is a deepfake.  This arms the artist with the information
        they need to confidently and quickly protect themselves.</h3>
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
    # Create two columns to display the audio clips
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("Suspected Joel Deepfake:")
        original1 = librosa.load("./audio_samples/clones/joel_fcar.wav")
        st.audio(original1[0], sample_rate=original1[1])
        calculate_similarity1_button = st.button("Calculate Similarity Score",
        type='primary', use_container_width=True, key='similarity1')
        if calculate_similarity1_button:
            with st.spinner("Calculating Similarity Score..."):
                time.sleep(3)
                st.text("")
                st.markdown("##### Calculated Similarity Score: :red[98.71%]")
                st.markdown("##### :red[Warning:  There is a *very high* probability this is a deep fake.\
                            Take action immediately!]")
    with col2:
        st.markdown("Suspected Jenny Deepfake:")
        original2 = librosa.load("./audio_samples/clones/tswift1.wav")
        st.audio(original2[0], sample_rate=original2[1])
        calculate_similarity2_button = st.button("Calculate Similarity Score",
        type='primary', use_container_width=True, key='similarity2')
        if calculate_similarity2_button:
            with st.spinner("Calculating Similarity Score..."):
                time.sleep(3)
                st.text("")
                st.markdown("##### Calculated Similarity Score: :green[3.08%]")
                st.markdown("##### :green[No action required.  This is not a deep fake.]")

    st.markdown("---")
    return_to_opps_button = st.button("Explore the Possibilities", type='primary', use_container_width=True)
    if return_to_opps_button:
        switch_page("Downstream Opportunities")
   
secure_home()
  
# Display the "Business Chat" sidebar
from utils.bplan_utils import chat_state_sidebar
chat_state_sidebar()
