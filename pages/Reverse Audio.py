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
                **Now that we have our voiceprints for Joel and Jenny established,**
                we can compare them to other audio files to see how similar they are.  This
                will help us determine if the audio file is Joel or Jenny.  If it is, then we can
                search the artist's issued NFTs to confirm if the counterparty has access to the
                voiceprint, and if not, we can proceed with issuing "cease and desist" or other
                legal action.
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
