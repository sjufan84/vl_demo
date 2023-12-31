""" Demo page for visualizing audio features """ 
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import soundfile as sf

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

def read_audio(audio_file):
    """ Read the audio file into a NumPy array
    using soundfile"""
    signal, _ = sf.read(audio_file)
    return (signal, _)


        
def demo_visualize():
    """ Demo page for visualizing audio features via Kmeans clustering """
    st.markdown("""<div class="text-container">
    <h4 style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 30px; font-weight: 550;">
    Melodic Voiceprint: A Harmony of Science, Art, and Security</h4>
    </div>""", unsafe_allow_html=True)
                
    st.markdown("""
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
        st.audio(joel_train_signal[0], sample_rate=joel_train_signal[1])
    with col2:
        st.markdown('**Jenny "B Major Scale"**')
        jenny_train_signal = read_audio('./audio_samples/jenny_train.wav')
        st.audio(jenny_train_signal[0], sample_rate = jenny_train_signal[1])
    st.markdown("""---""")
    continue_3d_button = st.button("Continue", type="primary", use_container_width=True)
    if continue_3d_button:
        switch_page("Voice Swaps")

if st.session_state.demo_visualize_page == "demo_visualize_home":
    demo_visualize()
