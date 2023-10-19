import streamlit as st
from streamlit_extras.switch_page_button import switch_page
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