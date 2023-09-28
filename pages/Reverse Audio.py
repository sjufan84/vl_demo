""" Reverse Audio page shown when the user enters the application """
import streamlit as st
import os
import numpy as np


def reverse_audio():
    """ Reverse Audio Homepage """
    st.markdown("""
                <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
                Reverse Audio Search
                </p>
                """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
                **One of the primary use cases for a secure voiceprint is\
                to identify potential copyright violations from generative content.\
                With the speed at which these deep fakes can be created, it is imperative\
                that the artist and other stakeholders have a quick and easy way to determine\
                whether or not the content is a) their voice, and b) has been approved for use.\
                By leveraging the high level math behind the voiceprint, we can quickly and\
                easily determine the similarity between two audio clips for further investigation\
                including the scan of the artist's issued NFTs to identify consent.**
                """)
    st.markdown("---")
    # Create two columns, one to demonstrate generating similarity scores for Joel's voiceprint
    # and the other for Jenny's.
    col1, col2 = st.columns(2, gap="large")
    with col1:
        # Load in the audio files from the "../audio_samples/comparison_clips/is_joel" directory
        # and create a list of their filenames.
        joel_audio_files = os.listdir("./audio_samples/comparison_clips/is_joel")
        joel_audio_files = ["./audio_samples/comparison_clips/is_joel/" + file for file in joel_audio_files]
        not_joel_audio_files = os.listdir("./audio_samples/comparison_clips/not_joel")
        not_joel_audio_files = ["./audio_samples/comparison_clips/not_joel/" + file for file in not_joel_audio_files]

        # Create a selectbox to choose which audio file to compare to Joel's voiceprint.
        joel_audio_file = st.selectbox("Choose a clip to compare to Joel's voiceprint", joel_audio_files + not_joel_audio_files)

        # If the clip is in the "is_joel" directory, generate and display a similarity score between 90 and 100.
        # Otherwise, generate and display a similarity score between 0 and 10.
        if joel_audio_file in joel_audio_files:
            st.markdown(f"#### Similarity Score: {np.random.randint(90, 100)}%")
        else:
            st.markdown(f"#### Similarity Score: {np.random.randint(0, 10)}%")

        # Display the audio clip.
        st.audio(joel_audio_file)

    with col2:
        # Load in the audio files from the "../audio_samples/comparison_clips/is_jenny" directory
        # and create a list of their filenames.
        jenny_audio_files = os.listdir("./audio_samples/comparison_clips/is_jenny")
        jenny_audio_files = ["./audio_samples/comparison_clips/is_jenny/" + file for file in jenny_audio_files]
        not_jenny_audio_files = os.listdir("./audio_samples/comparison_clips/not_jenny")
        not_jenny_audio_files = ["./audio_samples/comparison_clips/not_jenny/" + file for file in not_jenny_audio_files]

        # Create a selectbox to choose which audio file to compare to Jenny's voiceprint.
        jenny_audio_file = st.selectbox("Choose a clip to compare to Jenny's voiceprint", jenny_audio_files + not_jenny_audio_files)

        # If the clip is in the "is_jenny" directory, generate and display a similarity score between 90 and 100.
        # Otherwise, generate and display a similarity score between 0 and 10.
        if jenny_audio_file in jenny_audio_files:
            st.markdown(f"#### Similarity Score: {np.random.randint(90, 100)}%")
        else:
            st.markdown(f"#### Similarity Score: {np.random.randint(0, 10)}%")

        # Display the audio clip.
        st.audio(jenny_audio_file)

if __name__ == "__main__":
    reverse_audio()
