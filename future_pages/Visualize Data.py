""" This file contains the code for the Visualize Data page of the Streamlit app.
    We will leverage the functions from the audio_processing.py file to generate
    different visualizations of the audio data.  There will be data from the user's
    recordings as well as some dummy data for comparison. """
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import (
    load_audio, get_spectrogram,
    generate_mel_spectrogram, get_mfcc, 
    get_lfcc, get_pitch, plot_waveform, 
    plot_spectrogram, plot_pitch,
    play_audio
)


def plot_selected_feature(waveform, sample_rate, selection):
    """ Plot the selected feature and display its description. """
    plot_function, description = PLOT_OPTIONS[selection]
    st.markdown(f"### {selection}\n{description}")

    if selection == "Waveform":
        st.plotly_chart(plot_function(waveform, sample_rate))
    elif selection == "Spectrogram":
        specgram = get_spectrogram(waveform)
        st.plotly_chart(plot_function(specgram))
    elif selection == "Mel Spectrogram":
        melspec = generate_mel_spectrogram(waveform, sample_rate)
        st.plotly_chart(plot_function(melspec, title="Mel Spectrogram"))
    elif selection == "MFCC (Mel-Frequency Cepstral Coefficients)":
        mfcc = get_mfcc(waveform, sample_rate)
        st.plotly_chart(plot_function(mfcc, title="MFCC"))
    elif selection == "LFCC (Linear Frequency Cepstral Coefficients)":
        lfcc = get_lfcc(waveform, sample_rate)
        st.plotly_chart(plot_function(lfcc, title="LFCC"))
    elif selection == "Pitch":
        pitch = get_pitch(waveform, sample_rate)
        st.plotly_chart(plot_pitch(waveform, sample_rate, pitch))

# Define a dictionary to map the selection to a function and description
PLOT_OPTIONS = {
    "Waveform": (plot_waveform, "The waveform is a graphical representation of the audio\
                signal..."),
    "Spectrogram": (plot_spectrogram, "A spectrogram visualizes how the frequencies..."),
    "Mel Spectrogram": (plot_spectrogram, "Mel spectrogram represents frequencies in a way..."),
    "MFCC (Mel-Frequency Cepstral Coefficients)": (plot_spectrogram, "MFCCs describe\
    the overall shape of the sound spectrum..."),
    "LFCC (Linear Frequency Cepstral Coefficients)": (plot_spectrogram,
    "LFCCs are similar to MFCCs but are derived from a linear spacing..."),
    "Pitch": (plot_pitch, "Pitch is the perceived frequency of a sound..."),
}

def visualize_data():
    """ Visualize the audio data. """
    # Create a list of the audio files from the keys of the audio_files dictionary
    audio_files = list(st.session_state.audio_files.keys())

    st.markdown('#### Data Visualization')
    st.markdown('---')
    st.markdown('**By selecting an audio file from the dropdown menu below,\
                you can visualize various features...**')

    # Create a selectbox to choose the audio file
    selected_file = st.selectbox("Select an audio file", audio_files)

    # Load the audio data
    waveform, sample_rate = load_audio(selected_file)

    # Get the selection from the selectbox
    selection = st.selectbox("Choose the plot you want to see:", list(PLOT_OPTIONS.keys()))

    col1, col2 = st.columns([2, 1])

    # Plot the selected feature
    with col1:
        plot_selected_feature(waveform, sample_rate, selection)

    # Play the audio file
    with col2:
        st.markdown("""
        ### Play Audio:
                    """)
        st.write(play_audio(selected_file))
    



def no_audio():
    """ Display this page if the user has not uploaded any audio files. """
    st.warning('**You have not uploaded any audio files yet.  Please upload\
            at least one audio file before proceeding.\
            You may either upload a file from your local machine or record\
            audio using the Record Audio page.**')
    
    # Display buttons depending on whether or not the files were uploaded
    record_audio_button = st.button("Record Audio", type='primary', use_container_width=True)
    if record_audio_button:
        switch_page("Record Audio")
    upload_audio_button = st.button("Upload Audio", type='primary', use_container_width=True)
    if upload_audio_button:
        switch_page("Upload Audio")

# If the user has uploaded audio files, display the page
if st.session_state.audio_files:
    visualize_data()
else:
    no_audio()

