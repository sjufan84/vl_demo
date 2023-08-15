import streamlit as st
from PIL import Image


def minting_demo_page():
    # Introduction to NFT minting
    st.title("Understanding NFT Minting: A Demonstration")
    st.markdown("Minting an NFT (Non-Fungible Token) is like creating a digital certificate of authenticity that can be bought, sold, or traded. For artists, it's a way to ensure ownership and control over their work. Here's how it works:")

    # Step 1: Artist generates the NFT
    st.subheader("Step 1: Artist Generates the NFT")
    st.markdown("The artist starts by creating the NFT, associating it with\
                 their unique Melodic Voiceprint.  Once both parties have signed\
                the contract, the counterparty will then have access to the artist's\
                MV.")
    artist_name = st.text_input("Artist Name:")
    voiceprint_description = st.text_input("Voiceprint Description:")
    if st.button("Generate NFT"):
        st.success(f"NFT for '{artist_name}' generated successfully!")

    # Step 2: Artist sets a price
    st.subheader("Step 2: Set a Price for the NFT")
    st.markdown("The artist can set a price for the NFT in USD, defining its market value.")
    price = st.number_input("Price in USD:", min_value=1.00, step=0.01)
    if st.button("Set Price"):
        st.success(f"Price set to ${price} USD successfully!")

    # Step 3: Input contract metadata
    st.subheader("Step 3: Input Contract Metadata")
    st.markdown("The contract metadata includes terms, conditions, and any special agreements between the artist and the buyer.")
    contract_terms = st.text_area("Contract Terms:")
    if st.button("Input Metadata"):
        st.success("Contract metadata inputted successfully!")

    # Step 4: Both parties sign the contract
    st.subheader("Step 4: Both Parties Sign the Contract")
    st.markdown("Both the artist and the buyer (e.g., a label) need to sign the contract, agreeing to the terms.")
    artist_signature = st.checkbox("Artist Signature")
    buyer_signature = st.checkbox("Buyer/Label Signature")
    if artist_signature and buyer_signature:
        if st.button("Mint NFT"):
            st.success("NFT minted successfully!")
            display_nft_page(artist_name, voiceprint_description, price, contract_terms)


def display_nft_page(artist_name, voiceprint_description, price, contract_terms):
    st.title("Your Virtual NFT")
    st.markdown("Congratulations! You've minted a virtual NFT. Here's how it looks and what it represents:")

    # Display a cool visual (replace with the path to your image)
    nft_image_path = './nft_image.png'
    nft_image = Image.open(nft_image_path)
    st.image(nft_image, caption="Virtual NFT", use_column_width=True)

    # Display metadata
    st.subheader("NFT Metadata:")
    st.markdown(f"**Artist:** {artist_name}")
    st.markdown(f"**Voiceprint Description:** {voiceprint_description}")
    st.markdown(f"**Price:** ${price} USD")
    st.markdown(f"**Contract Terms:** {contract_terms}")

    # Explain how NFT relates to voiceprint
    st.subheader("How the NFT Relates to the Voiceprint:")
    st.markdown("The NFT is a digital representation of the artist's unique voiceprint. It carries the information and ownership rights, ensuring authenticity and control.")

    # Explain secure storage
    st.subheader("Secure Storage:")
    st.markdown("Your NFT is stored securely on a virtual blockchain platform. In a real-world scenario, it would be encrypted and stored on a decentralized network, ensuring protection and immutability.")

# You can call this function with the appropriate parameters once the NFT is minted


# Call the function to display the page
minting_demo_page()
