""" Demo page for visualizing audio features """ 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import librosa
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Voice Lockr Demo", page_icon=":microphone:",
                initial_sidebar_state="collapsed", layout="wide")

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'demo_visualize_page'
    ]
    default_values = [
        'demo_visualize_home'
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    


        
def demo_visualize():
    """ Demo page for visualizing audio features via Kmeans clustering """
    st.markdown("""
    ### :blue[Melodic Voiceprint: A Harmony of Science, Art, and Security]
                
    **The logical first question to ask is:**  How do these deepfake audio clips work,\
    and what can artists do to protect themselves?  The answer lies in securing the\
    Melodic Voiceprint of the artist that is being used to train the models\
    that make these deepfakes possible.

    **First let's establish a ground truth for each of our models.  Both Jenny and Joel's models were trained\
                on *less than 10 minutes* of clips of their vocals.  We can now swap out their voices for another
                artist's, even though neither Joel nor Jenny sang any part of the songs we used in this demo.
                Below are a few of the original clips that were used to train the models.**
    """)
    st.text("")
    # Create two columns, one for Joel's audio and one for Jenny's.
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('**Joel "Happy Birthday"**')
        # Convert the clip to a useable format
        joel_train_signal = read_audio('./audio_samples/joel_train.wav')
        st.audio(joel_train_signal, sample_rate=16000)
    with col2:
        st.markdown('**Jenny "B Major Scale"**')
        jenny_train_signal = read_audio('./audio_samples/jenny_train.wav')
        st.audio(jenny_train_signal, sample_rate=16000)
    st.markdown("""---""")
    continue_3d_button = st.button("Continue", type="primary", use_container_width=True)
    if continue_3d_button:
        switch_page("Voice Swaps")

    st.markdown("""**Now let's see what happens when we swap out the voices of Joel and Jenny for another artist's.
                There are two clips we have used for this demo.  The first is Luke Combs's rendition of "Fast Car", 
                and the other is Ella Henderson's "Ghost".  We can swap out male for female voices
                by utilizing the same methodology of other voice clones and simply adjust the pitch to create the best
                version.  This proves that this technology is here today, evolving, and constantly getting more 
                powerful.**""")
    # Create a selectbox to allow the user to select which song to visualize
    song = st.selectbox("Select a Song to Visualize:", ["Fast Car", "Ghost"])
    if song == "Fast Car":
        get_3d_chart_fcar()
    else:
        get_3d_chart_ghost()
            
    st.markdown("""---""")
    st.markdown("""
                 **So What Does This Mean for Music, Security, and the Future of the Industry?**

    1. **Understanding the Voice**: By analyzing these features, we can create a "Melodic Voiceprint,"
                a unique signature of an artist's voice. It's like a fingerprint for their voice,
                capturing the subtle nuances that make their voice distinctly theirs.

    2. **Protecting Authenticity**: The Melodic Voiceprint can be used to determine whether a piece of
        audio is genuinely from the claimed artist or not. It's a powerful tool to detect deepfakes,
        which are artificially created audio files that convincingly imitate a real artist's voice.

    3. **Application in Music**: For musicians, the Melodic Voiceprint
        safeguards artistic integrity. It ensures that their creative work
        remains authentic and unaltered, protecting against potential deepfake manipulation.

    4. **A New Layer of Security**: In the digital age, where voices can be forged,
        the Melodic Voiceprint acts as a cutting-edge solution to maintain the authenticity of vocal identity.
                
    5. **Downstream Possibilities**:
    * **Content Generation**: The Melodic Voiceprint can be utilized to develop personalized content, such as custom music, voiceovers, and more.
    * **Voice Authentication**: It offers a robust method for voice-based authentication in security systems.
    * **Enhanced Creativity**: Musicians and creators can experiment with voice manipulation, remixing, and other artistic expressions while preserving authenticity.
    * **New Business Models**: The Melodic Voiceprint can be used to create new revenue streams for artists, such as personalized content and voice authentication.
    """)

    st.text("")
    st.markdown("""
                **By securing the Melodic Voiceprint through NFTs**, or non-fungible tokens,
                Vocalockr ensures unique and protected ownership. An NFT represents a binding
                contract between an artist and an owner, whether a record label, streaming
                service, or fan. Without owning the NFT, usage of the artist's voice is
                unapproved. This method not only safeguards the artist's voice but also
                guarantees that it's used in line with their wishes, offering a powerful
                tool in the evolving digital landscape of music.
                """)
    st.text("")
    mint_nft_button = st.button("Mint an MV NFT", type="primary", use_container_width=True)
    if mint_nft_button:
        st.session_state.nft_demo_page = "nft_demo_home"
        switch_page("Generate NFT")


if st.session_state.demo_visualize_page == "demo_visualize_home":
    demo_visualize()
