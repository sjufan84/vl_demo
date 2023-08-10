""" This file contains the code for the Visualize Data page of the Streamlit app.
    We will leverage the functions from the audio_processing.py file to generate
    different visualizations of the audio data.  There will be data from the user's
    recordings as well as some dummy data for comparison. """
from io import StringIO
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.audio_processing import load_audio, get_spectrogram, generate_mel_spectrogram, get_mfcc, get_lfcc, get_pitch, plot_waveform, plot_spectrogram, plot_pitch

def visualize_data():
    """ Visualize the audio data. """
    # Create a list of the audio files from the keys of the audio_files dictionary
    audio_files = list(st.session_state.audio_files.keys())

    st.markdown('#### Data Visualization')
    st.markdown('---')
    st.markdown('**By selecting an audio file from the dropdown menu below, you can visualize\
                various features of the audio data.  These features are what comprises your unique Melodic\
                Voiceprint.**')

    # Create a selectbox to choose the audio file
    selected_file = st.selectbox("Select an audio file", audio_files)

    # Load the audio data
    waveform, sample_rate = load_audio(selected_file)

    # Define the options for the selectbox
    options = [
        "Waveform",
        "Spectrogram",
        "Mel Spectrogram",
        "MFCC (Mel-Frequency Cepstral Coefficients)",
        "LFCC (Linear Frequency Cepstral Coefficients)",
        "Pitch",
    ]

    # Create the selectbox
    selection = st.selectbox("Choose the plot you want to see:", options)

    # Show the plot and description based on the selection
    if selection == "Waveform":
        st.markdown("""
        ### Waveform
        The waveform is a graphical representation of the audio signal. It shows the amplitude of the sound wave at each point in time. The x-axis represents time, and the y-axis represents the amplitude of the wave.
        """)
        st.plotly_chart(plot_waveform(waveform, sample_rate))

    elif selection == "Spectrogram":
        st.markdown("""
        ### Spectrogram
        A spectrogram visualizes how the frequencies present in the sound are spread across time. The x-axis represents time, the y-axis represents frequency, and the color represents the amplitude of a particular frequency at a given time.
        """)
        specgram = get_spectrogram(waveform)
        st.plotly_chart(plot_spectrogram(specgram))

    elif selection == "Mel Spectrogram":
        st.markdown("""
        ### Mel Spectrogram
        Mel spectrogram represents frequencies in a way that's closer to how the human ear perceives sound. The x-axis represents time, the y-axis represents Mel frequency bins, and the color represents the amplitude of a particular frequency at a given time.
        """)
        melspec = generate_mel_spectrogram(waveform, sample_rate)
        st.plotly_chart(plot_spectrogram(melspec, title="Mel Spectrogram"))

    elif selection == "MFCC (Mel-Frequency Cepstral Coefficients)":
        st.markdown("""
        ### MFCC
        MFCCs describe the overall shape of the sound spectrum, often used for speech and audio processing. The x-axis represents time, the y-axis represents the order of coefficients, and the color indicates the value of each coefficient.
        """)
        mfcc = get_mfcc(waveform, sample_rate)
        st.plotly_chart(plot_spectrogram(mfcc, title="MFCC"))

    elif selection == "LFCC (Linear Frequency Cepstral Coefficients)":
        st.markdown("""
        ### LFCC
        LFCCs are similar to MFCCs but are derived from a linear spacing of frequencies. The x-axis represents time, the y-axis represents the order of coefficients, and the color indicates the value of each coefficient.
        """)
        lfcc = get_lfcc(waveform, sample_rate)
        st.plotly_chart(plot_spectrogram(lfcc, title="LFCC"))

    elif selection == "Pitch":
        st.markdown("""
        ### Pitch
        Pitch is the perceived frequency of a sound. The x-axis represents time, and the y-axis represents the pitch measured in Hz. This plot can be used to understand the variations in pitch throughout the audio.
        """)
        pitch = get_pitch(waveform, sample_rate)
        st.plotly_chart(plot_pitch(waveform, sample_rate, pitch))


def no_audio():
    """ Display this page if the user has not uploaded any audio files. """
    st.warning('**You have not uploaded any audio files yet.  Please upload at least one audio file before proceeding.\
               You may either upload a file from your local machine or record audio using the Record Audio page.**')
    
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

