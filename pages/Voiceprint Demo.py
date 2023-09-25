""" Here we will demonstrate the voiceprint functionality of the app
by using a 3d plot of Joel and Jenny's voiceprints. """

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def voiceprint_visual():
    """ Voiceprint Homepage """
    st.markdown("""<div class="text-container">
    <h4 style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
                So Now What?</h4>
    </div>""", unsafe_allow_html=True)
    st.markdown('##### Now that we have our voiceprints for Joel and Jenny established,\
                what can we do with them?  The first order of business is to secure them\
                for the artist so that they can decide what features they would like to explore.\
                Once we have that taken care of, if the artist chooses, we have multiple opportunities\
                to use the voiceprint downstream.  These include, but our not limited to,\
                "Reverse Audio Search", "Co-writer", and other ways to issue contracts as NFTs\
                for trackability, security, and peace of mind.')
    st.text("")
    # Create buttons for the various use cases
    nft_button = st.button("Issue NFT", type = 'primary', use_container_width=True)
    reverse_audio_button = st.button("Reverse Audio Search", type = 'primary', use_container_width=True)
    co_writer_button = st.button("Co-writer", type = 'primary', use_container_width=True)

    # If the user clicks the "Issue NFT" button, switch to the NFT Demo page.
    if nft_button:
        switch_page("Generate NFT")
    if reverse_audio_button:
        switch_page("Reverse Audio")
    if co_writer_button:
        switch_page("Co-writer")


    
if __name__ == "__main__":
    voiceprint_visual()