""" This module contains the code for the NFT minting demo. """
import os
from dotenv import load_dotenv
import streamlit as st
from web3 import Web3

# Load the environment variables
load_dotenv()

# Connect to Ethereum (Replace with your provider URL)
w3 = Web3(Web3.HTTPProvider(f"{os.getenv('INFURA_KEY')}"))
st.session_state.w3 = w3
# Contract ABI (Replace with the actual ABI of your contract
# Read in the contract ABI
with open('./contracts/compiled/mv_contracts.json', 'r') as f:
    contract_abi = f.read()

# Contract Address (Replace with the actual address of your deployed contract)
contract_address = F'{os.getenv("CONTRACT_ADDRESS")}'

# Create Contract Instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'nft_demo_page', 'token_id', 'artist_signed', 'nft_metadata', 'label_wallet', 'receipt'
    ]
    default_values = [
        'nft_demo_home', 0, False, {}, '', {}
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_variables()

def get_metadata(token_id):
    """ Fetches the metadata of the minted NFT. """
    # Example: Fetching the metadata of the minted NFT (Replace with actual fetching)
    metadata = contract.functions.getMetadata(token_id).call({'from': os.getenv('ARTIST_WALLET')})
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
    # Load the artist wallet
    artist_wallet = os.getenv("ARTIST_WALLET")
    label_wallet = os.getenv("LABEL_WALLET")
    with st.form("minting_form", clear_on_submit=False):
        value = st.number_input("Enter the value of the contract:", value=10000, step=1)
        contract_verbiage = st.text_input("Enter the verbiage of the contract:")
        reason = st.text_input("Enter the reason for the NFT:")
        approved_artist_address = artist_wallet
        artist_address = st.selectbox("Select your wallet address:", [artist_wallet, "0x1234", "0x5678"])
        artist_private_key = st.text_input("Enter your private key:", type='password', value = os.getenv("ARTIST_PRIVATE_KEY"))
        # Select the wallet address of the label
        label_address = st.selectbox("Select the wallet address of the label:", [label_wallet, "0x1234", "0x5678"])
        # Set an approved label address
        approved_label_address = label_wallet
        # Create a button to mint the NFT
        mint_button = st.form_submit_button("Sign and Initiate the NFT", type='primary', use_container_width=True)
        if mint_button:
            if artist_address != approved_artist_address or label_address != approved_label_address:
                st.error("You are not authorized to mint this NFT\
                         or the label address is not approved\
                        . Please select an approved wallet address.")
                st.stop()
            # Set the metadata for the NFT
            #metadata = {
            #    "value": value,
            #    "contract_verbiage": contract_verbiage,
            #    "reason": reason,
            #    "mv_link": "http://someserver.com/voiceprint",
            #    "artist_wallet": artist_wallet,
            #    "label_wallet": label_wallet,
            #}
            # Dummy link for the MV
            mv_link = "http://someserver.com/voiceprint"
            # Connect to the contract        
            # Call the initiateNFT function from the contract (Replace with correct function name and parameters)
            tx = contract.functions.initiateNFT(value, contract_verbiage, reason, mv_link, label_wallet).build_transaction(
                {
                    'from': artist_wallet,
                    'gas': 1000000,
                    'nonce': w3.eth.get_transaction_count(artist_wallet)
                }
            )
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=os.getenv("ARTIST_KEY"))
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            rich_logs = contract.events.NFTInitiated().process_receipt(receipt)
            token_id = rich_logs[0]['args']['tokenId']
            st.session_state.latest_nft = token_id
            # Display a success message with the transaction hash
            st.success(f"Successfully initiated NFT # {token_id}!\
                        \nTransaction Hash: {tx_hash.hex()}")
            
            st.session_state.artist_signed = True
    
    if st.session_state.artist_signed:
        # Create a button to go to the next step
        proceed_button = st.button("Proceed to Label Signing Demo", type='primary', use_container_width=True)
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
    label_wallet = os.getenv("LABEL_WALLET")
    st.markdown(" ##### For the purposes of the demo, we will automatically\
                import the artist's contract and metadata.  In further\
                deployments record labels could have many more options\
                including search for available artists, etc.")
    st.markdown('---')
    # Display available NFTs
    st.markdown(f"\
        ##### The artist if offering the following NFT :blue[#{st.session_state.latest_nft}]\
        for your signature.\
        Please review the details and sign if you agree to the terms.  The\
        Metadata is listed below:\
               ")
    # Send a transaction to get the metadata
    metadata = get_metadata(st.session_state.latest_nft)

    st.markdown("#### :blue[Contract Metadata:]")
    # Display details of selected NFT
    #st.write(f"**Contract Value:** {metadata['value']}")
    #st.write(f"**Contract Verbiage:** {metadata['contract_verbiage']}")
    #st.write(f"**Voice Print Link:** {metadata['mv_link']}")
    #st.write(f"**Reason:** {metadata['reason']}")
    #st.write(f"**Artist Signed:** {metadata['artist_wallet']}")
    #tx_hash = w3.eth.wait_for_transaction_receipt(metadata)['transactionHash'].hex()
    #st.write(f"Transaction Hash: {tx_hash}")
                                                
    # Display details of selected NFT
    
    # Additional details, images, etc., can be added here
    st.write(f"**Contract Value:** {metadata[0]}")
    st.write(f"**Contract Verbiage:** {metadata[1]}")
    st.write(f"**Voice Print Link:** {metadata[2]}")
    st.write(f"**Reason:** {metadata[3]}")
    st.write(f"**Artist Signed:** {metadata[4]}")
    st.write(f"**Counterparty Signed:** {metadata[5]}")
    # Allow the user to select their wallet address
    wallet_address = st.selectbox("Select your wallet address:", [label_wallet, "0x1234", "0x5678"])
    approved_wallet_address = label_wallet
    # Signing process
    if st.markdown("If the contract looks good, you may click to sign the contract below.\
                 Remember, once it is on the Blockchain it **cannot be reversed**,\
                 so ensure everything is correct before approving."):
        st.warning("#### @Joel -- this is as far as I've gotten.  At this point the artist is\
                   interacting with the actual contract that I am hosting locally, but it will need\
                   to be flipped for the actual deployment.")
        # Create a button to sign the contract
        sign_button = st.button("Sign Contract", type='primary', use_container_width=True)
        if sign_button:
            if wallet_address != approved_wallet_address:
                st.error("You are not authorized to sign this contract. Please select an approved wallet address.")
                st.stop()
            # Call the signNFT function from the contract
            tx = contract.functions.counterpartySign(st.session_state.latest_nft).build_transaction(
                {
                    "from": label_wallet,
                    'gas': 1728712,
                    "nonce": w3.eth.get_transaction_count(label_wallet),
                }
            )
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=os.getenv("LABEL_KEY"))
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            logs = contract.events.NFTSigned().process_receipt(receipt)
            st.write(logs)
            st.session_state.label_signed = True

            # Display a success message with the transaction hash
            st.success(f"Successfully signed NFT # {st.session_state.latest_nft}!\
                       Waiting on artist to sign contract...")
            st.session_state.label_signed = True


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
