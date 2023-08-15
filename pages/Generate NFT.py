import streamlit as st
from web3 import Web3

# Connect to Ethereum (Replace with your provider URL)
def conntect_to_contract():
    """ Connect to the contract and return the contract instance."""
    w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:7545"))

    # Contract ABI (Replace with the actual ABI of your contract)
    contract_abi = []

    # Contract Address (Replace with the actual address of your deployed contract)
    contract_address = "THIS IS A PLACEHOLDER ADDRESS"

    # Create Contract Instance
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)

    return contract

def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'nft_demo_page', 'token_id', 'artist_signed'
    ]
    default_values = [
        'nft_demo_home', 0, False
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_variables()

def get_metadata(token_id):
    """ Fetches the metadata of the minted NFT. """
    contract = conntect_to_contract()
    # Example: Fetching the metadata of the minted NFT (Replace with actual fetching)
    metadata = contract.functions.getMetadata(token_id).call()
    return metadata

    
def minting_demo_home():
    """ Home page for the NFT minting demo. Allows the user
    to emulate the process of minting the NFT as an artist would."""
    # Introduction to NFT minting
    st.markdown("### :blue[Understanding NFT Minting: A Demonstration]")
    st.markdown("""**Minting an NFT (Non-Fungible Token) is like
                creating a digital certificate of authenticity that
                can be bought, sold, or traded. For artists, it's
                a way to ensure ownership and control over their work.
                Here's how it works:**""")

    # Step 1: Artist generates the NFT
    st.subheader(":blue[Step 1: Artist Generates the NFT]")
    st.markdown("""**The artist starts by creating the NFT, associating it with
                 their unique Melodic Voiceprint.  Once both parties have signed
                the contract, then and only then will the counterparty have access
                to the MV, and only on the terms in which the artist has agreed to.
                We will walk through each step in the minting process to see how it works.**""")
    
    # Create a button to go to the next step
    artist_demo_button = st.button("Proceed to Artist Minting Demo",
                                    type='primary', use_container_width=True)
    if artist_demo_button:
        st.session_state.nft_demo_page = "artist_minting_demo"
        st.experimental_rerun()

def artist_minting_demo():
    """ Allows the user to emulate the process of minting
    the NFT as an artist would."""
    st.warning("""
               **@Joel: This page is as far as I've gotten.  The
               smart contract has basically been written, it's just a matter
               of deploying it to a testnet.  Because this will require actual
               demo wallet addresses, the integration will require a few
               more steps.**""")
    # Artist Actions
    st.markdown("### :blue[Artist NFT Demo]")
    st.text("")
    st.markdown("""
                ##### In line with our mission to serve artists first,\
                the power to initialize rests entirely with the artist\
                and any other approved (by the artist) parties.\
                The Vocalockr team will never initiate an NFT on behalf of an artist\
                unless explicitly requested to do so.  Below you can walk through\
                a very basic representation of the process of minting an NFT.\
                """)
    st.text("")
    # We will create a form to take in all of the relevant information
    # for minting the NFT
    with st.form("minting_form", clear_on_submit=True):
        value = st.number_input("Enter the value of the contract:")
        contract_verbiage = st.text_input("Enter the verbiage of the contract:")
        reason = st.text_input("Enter the reason for the NFT:")
        approved_wallet_address = "0x123"
        wallet_address = st.selectbox("Select your wallet address:", ["0x123", "0x456", "0x789"])
        # Create a button to mint the NFT
        mint_button = st.form_submit_button("Sign and Mint NFT", type='primary', use_container_width=True)
        if mint_button:
            if wallet_address != approved_wallet_address:
                st.error("You are not authorized to mint this NFT. Please select an approved wallet address.")
                st.stop()
            # Connect to the contract
            contract = conntect_to_contract()
            # Call the initiateNFT function from the contract (Replace with correct function name and parameters)
            txn_hash = contract.functions.initiateNFT(wallet_address, value, contract_verbiage, reason).transact()
            # Display a success message with the transaction hash
            st.success(f"Transaction Hash: {txn_hash}")
            st.session_state.artist_signed = True
    
    if st.session_state.artist_signed:
        # Create a button to go to the next step
        proceed_button = st.button("Proceed to Label Signing Demo")
        if proceed_button:
            st.session_state.nft_demo_page = "label_signing_demo"
            st.experimental_rerun()
        

    # Function to Set Developer's Address
    #if st.button("Set Developer's Address"):
    #    developer_address = st.text_input("Enter the developer's address:")
        
        # Call the setDeveloperAddress function from the contract
    #    txn_hash = contract.functions.setDeveloperAddress(developer_address).transact()
    #    st.write(f"Transaction Hash: {txn_hash}")

    # Function to Get Metadata
    #if st.button("Get Metadata"):
    #    token_id = st.number_input("Enter the token ID:")
        
        # Call the getMetadata function from the contract
    #    metadata = contract.functions.getMetadata(token_id).call()
    #    st.write(f"Metadata: {metadata}")


def label_interface():
    """ Allows the user to emulate the process of signing
      the NFT as a record label would."""
    st.markdown("### NFT Record Label Simulator")
    st.markdown('---')
    st.markdown("""
                **For the purposes of the demo, we will automatically
                import the artist's contract and metadata.  In further 
                deployments record labels could have many more options
                including search for available artists, etc.**
                """)

    # Example: Fetching available NFTs for signing (You'll need to replace this with actual data fetching)
    available_nfts = [{"tokenId": 1, "artist": "Artist 1", "reason": "Album Release"}, 
                      {"tokenId": 2, "artist": "Artist 2", "reason": "Single Release"}]

    # Display available NFTs
    selected_nft = st.selectbox("Select an NFT to view details:", available_nfts,
                            format_func=lambda x: f"Token ID: {x['tokenId']} - Artist: {x['artist']} - Reason: {x['reason']}")

    # Display details of selected NFT
    st.subheader("NFT Details:")
    st.write(f"Token ID: {selected_nft['tokenId']}")
    st.write(f"Artist: {selected_nft['artist']}")
    st.write(f"Reason: {selected_nft['reason']}")

    # Additional details, images, etc., can be added here

    # Signing process
    if st.button("Sign NFT"):
        confirm = st.button("Are you sure you want to sign this NFT?")
        if confirm:
            # Call the counterpartySign function from the contract
            # ...
            st.success("NFT successfully signed!")

def display_nft_metadata():
    """ Displays the metadata of the NFT."""
    metadata = get_metadata(st.session_state.token_id)
    # Display Metadata
    st.subheader("NFT Details:")
    st.write(f"Value: {metadata['value']}")
    st.write(f"Contract Verbiage: {metadata['contractVerbiage']}")
    st.write(f"Voice Print Link: {metadata['voicePrintLink']}")
    st.write(f"Reason: {metadata['reason']}")
    st.write(f"Artist Signed: {metadata['artistSigned']}")
    st.write(f"Counterparty Signed: {metadata['counterpartySigned']}")

# Call the function to display the page
if st.session_state.nft_demo_page == "nft_demo_home":
    minting_demo_home()
elif st.session_state.nft_demo_page == "artist_minting_demo":
    artist_minting_demo()
elif st.session_state.nft_demo_page == "label_signing_demo":
    label_interface()
elif st.session_state.nft_demo_page == "display_nft_metadata":
    display_nft_metadata()
