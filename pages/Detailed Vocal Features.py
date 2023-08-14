""" A Glossary of terms used in the application."""
import streamlit as st
from PIL import Image
from utils.audio_processing import (
    load_audio, get_spectrogram,
    plot_waveform, plot_spectrogram,
    generate_mel_spectrogram,
    get_pitch, plot_pitch
)

# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        "detailed_vocal_features_page"
    ]
    default_values = [
        "detailed_home"
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()    

st.markdown("""### Detailed Vocal Features""")
st.markdown("---")

st.markdown("""##### Select a feature below to visualize.\
            These features are extracted from the audio samples\
            and are commonly used to identify traits of the vocalists\
            as well as train the types of machine learning models that\
            generate deep fake audio.  They also partially make up the\
            Melodic Voiceprint that is used to protect the artist from\
            such vocal clones.""")

labels = ["Aloe Blacc", "Luke Combs"]  # Labels for the audio clips

def detailed_home_page():
    """ Demo page for visualizing audio features """
    audio_files = ['./audio_samples/avicii1.wav', './audio_samples/combs1.wav']
    
    
    # Load and process audio files
    audio_waveforms = [load_audio(file) for file in audio_files]

    # Select feature to visualize
    feature_option = st.selectbox(
        "Select a feature to visualize:",
        ("Waveform", "Spectrogram", "Mel Spectrogram", "LFCC", "Pitch", "KMeans Clustering")
    )

    for i, (waveform, sr) in enumerate(audio_waveforms):
        plot_feature(waveform, sr, feature_option, audio_files[i], labels[i])

def load_plots():
    """ Load the relevant plots instead of generating them each time for faster performance """
    # Create a list of feature options
    feature_options = ["Waveform", "Spectrogram", "Mel Spectrogram", "Pitch"]
    feature_option = st.selectbox(
        "Select a feature to visualize:",
        feature_options
    )

    audio_files = ['./audio_samples/avicii1.wav', './audio_samples/combs1.wav']
    plot_filenames = {
        "Waveform": ['./plots/aloe_waveform.png', './plots/combs_waveform.png'],
        "Spectrogram": ['./plots/aloe_spectrogram.png', './plots/combs_spectrogram.png'],
        "Mel Spectrogram": ['./plots/aloe_mel_spect.png', './plots/combs_mel_spect.png'],
        "Pitch": ['./plots/aloe_pitch.png', './plots/combs_pitch.png']
    }
    plot_labels = ["Aloe Blacc", "Luke Combs"]  # Labels for the audio clips
    # Descriptions for each feature
    feature_descriptions = {
        "Waveform": "A visual representation of a sound signal, showing the changes\
        in amplitude over time. It's like a snapshot of the ups and downs of a sound wave.",
        "Spectrogram": "A graphical representation that shows how different frequencies are\
        present in a sound signal over time. Imagine it as a colorful landscape of the sound,\
        where different colors represent different frequencies.",
        "Mel Spectrogram": "A special type of spectrogram that reflects the way humans perceive\
        sound. It's a map that can help to understand what parts of the sound are more noticeable\
        to the human ear.",
        "Pitch": "A visual display of the perceived frequency of a sound, often related to the\
        musical note. Think of it as showing the highness or lowness of the sound's musical tone."
    }
    # Display the feature description
    st.markdown(f"**:blue[{feature_descriptions[feature_option]}]**")

    for audio_file, plot_file, plot_label in zip(audio_files, plot_filenames[feature_option], plot_labels):
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Set the plot title
            st.markdown(f"**{plot_label} {feature_option}**")
            # Load the relevant plot
            plot_image = Image.open(plot_file)
            st.image(plot_image, use_column_width=True)

        with col2:
            st.markdown("**Audio Clip**")
            # Display the audio clip
            col2.audio(audio_file, format='audio/wav')


def plot_feature(waveform, sr, feature_option, audio_file, label):
    """ Plot the selected feature """
    col1, col2 = st.columns([2, 1])  # Adjust ratios as needed
    with col1:
        st.markdown(f"**{label} - {feature_option}**")  # Add the label and title
        if feature_option == "Waveform":
            col1.plotly_chart(plot_waveform(waveform, sr))
        elif feature_option == "Spectrogram":
            col1.plotly_chart(plot_spectrogram(get_spectrogram(waveform)))
        elif feature_option == "Mel Spectrogram":
            col1.plotly_chart(plot_spectrogram(generate_mel_spectrogram(waveform, sr)))
        elif feature_option == "Pitch":
            col1.plotly_chart(plot_pitch(waveform, sr, get_pitch(waveform, sr)))
    with col2:
        st.markdown(f"**{label} - Audio Clip**")  # Add the label for the audio clip
        col2.audio(audio_file, format='audio/wav')

if st.session_state.detailed_vocal_features_page == "detailed_home":
    load_plots()