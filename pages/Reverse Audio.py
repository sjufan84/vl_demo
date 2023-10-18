""" Reverse Audio page shown when the user enters the application """
import time
import streamlit as st
import librosa
from streamlit_extras.switch_page_button import switch_page


def reverse_audio():
    """ Reverse Audio Homepage """
    st.markdown("""
                <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
                Reverse Audio Search
                </p>
                """, unsafe_allow_html=True)
    st.markdown("""
                One of the primary use cases for a secure voiceprint is\
                to identify potential copyright violations from generative content.\
                With the speed at which these deep fakes can be created, it is imperative\
                that the artist and other stakeholders have a quick and easy way to determine\
                whether or not the content is a) their voice, and b) has been approved for use.\
                By leveraging the high level math behind the voiceprint, we can quickly and\
                easily determine the similarity between two audio clips for further investigation\
                including the scan of the artist's issued NFTs to identify consent.
                """)
    st.text("")
    st.markdown("""
                Once we have the artist's model trained, they will be able to 
                upload any audio clip that they suspect is a deep fake, and we will
                be able to generate a "similarity score" that indicates the probability
                that the clip is a clone.  By doing so, we can arm the artist with the tools
                they need to take further action if necessary.  We illustrate a few examples
                below.
                """)
    st.text("")
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
                st.markdown("##### Calculated Similarity Score: :red[98%]")
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
                st.markdown("##### Calculated Similarity Score: :green[3%]")
                st.markdown("##### :green[No action required.  This is not a deep fake.]")

    st.markdown("---")
    return_to_opps_button = st.button("Explore More Features", type='primary', use_container_width=True)
    if return_to_opps_button:
        switch_page("Voiceprint Demo")

    
      
    # Create two columns, one to demonstrate generating similarity scores for Joel's voiceprint
    # and the other for Jenny's.
   

if __name__ == "__main__":
    reverse_audio()
