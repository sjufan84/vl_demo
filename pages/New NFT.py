""" NFT Protocol Intro Page"""
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.markdown("""
<div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 26px; font-weight: 550; animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">The Artist Vault Protocol</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">We believe in a hybrid approach to securing
    our artists' voiceprints.  By leveraging the security of private blockchain technology
    while simultaneously providing the transparency of public blockchain verification, we
    can ensure that the artist's voiceprint is protected and secure while still allowing
    for public verification of the licensing agreements.  This will also allow us to abstract away
    the more technical aspects of blockchain technology, allowing artists to focus on what they do best:
    creating art.  Below we outline the steps of the Artist Vault Protocol:
    </h3>
</div>
   <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 1: Artist Onboarding
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">
    We begin by helping the artist train their unique voiceprint model.  This is done by
    working with the artist to select high quality vocal data that can be used to generate 
    the highest quality model possible.
    </h3>
    <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 2: Securing Data with Hybrid Blockchain Technology
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Upon onboarding, the artist's voiceprint
    is encrypted and recorded on a private blockchain, along with their custom trained models, 
    akin to placing a valuable item in a safe deposit box. This "Artist Vault" blockchain is managed by AWS, providing
    a robust and secure environment that supports fiat transactions, ensuring
    artists don't need to manage or transact in cryptocurrency.</h3>
    <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 3: Licensing Contracts
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Artists have the ability to license
    their voiceprints for various uses through a contractual agreement. Each contract
    specifies the terms of use, compensation, and any limitations. Contracts are digitally
    signed and stored within the Artist Vault, ensuring only authorized parties
    have access to the full terms and conditions.  Any downstream
    use cases are monitored for <i>credit, compensation, and consent,</i> with custom watermarks
    and other security measures to ensure the artist's voiceprint is protected along the way.
</h3>
    <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 4: Fiat Transactions on
    the Private Blockchain
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">When a voiceprint is licensed,
    the financial transaction is conducted in fiat currency on the private blockchain.
    This process is seamless for the artist and the licensing party, with our platform
    automating the handling of these transactions securely, reflecting a traditional
    payment experience without the need for cryptocurrency understanding.  
    </h3>
<div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 5: Revenue and Fees
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">For every contract issued,
    First Rule automatically deducts a 5% fee from the contract's total value.
    This fee is for platform maintenance, providing the service, and ensuring
    continuous improvement of the Artist Vault. The remainder of the contract's
    value is disbursed to the artist as per the agreed terms.
</h3>
<div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Step 6: NFT Minting for Public Representation
</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Once a contract is established
    and the voiceprint is securely stored, an NFT representing the contract's terms
    is minted on the public Ethereum blockchain. This NFT serves as a transparent and
    verifiable certificate of the licensing agreement, visible to anyone for confirmation
    and trust in the contract's validity, while the sensitive details remain protected on
    the private blockchain.
</h3>
<div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 20px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Conclusion</h4>
</div>
<div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">The Artist Vault elegantly merges
    the security of private blockchain technology with the transparency of public blockchain
    verification. This hybrid approach positions our platform at the forefront of digital
    rights management and monetization in the creative industry.</h3>
</div>

</div>
<style>
    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
            -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
            -o-animation: fadeIn ease 3s; -ms-animation:
            fadeIn ease 3s;">
            </div>""", unsafe_allow_html=True)
st.text("")

artist_vault_button = st.button("Voiceprint Licensing Demo", type = 'primary', use_container_width=True)
if artist_vault_button:
    switch_page("Generate NFT")

co_writer_button = st.button("Try Out Co-writer", type = 'primary', use_container_width=True)
if co_writer_button:
    switch_page("Co-writer")

back_to_home_button = st.button("Back to Home", type = 'primary', use_container_width=True)
if back_to_home_button:
    switch_page("main")
