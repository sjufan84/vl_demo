""" The main page to take the inputted .wav files and process them. 
We will use SpeechBrain's pretrained model to extract the embeddings
and then map them to a 3d space using UMAP.  We will then use the
embeddings to create the NFT. """

import streamlit as st
#from speechbrain.pretrained import EncoderDecoderASR

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'process_page'
    ]
    default_values = [
        'view_recordings'
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()

def display_current_recordings():
    """ Display the current recordings in a selectbox for the user to be able to
    view and / or adjust before submitting for processing."""
    # Display the current recordings in a selectbox for the user to be able to
    # view and / or adjust before submitting for processing.
    st.markdown('''<div style="text-align: center;">
                <h3 class="title">Current Recordings</h3>
                </div>''', unsafe_allow_html=True)
    st.text('')
    current_recordings = st.selectbox("Select a recording to view", list(st.session_state.samples.keys()))
    st.write("You selected: ", current_recordings)




#asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
#asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav')

if st.session_state.process_page == 'view_recordings':
    display_current_recordings()
