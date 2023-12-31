""" This module contains the code for the NFT minting demo. """
import os
import time
from dotenv import load_dotenv
import uuid
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Load the environment variables
load_dotenv()


# Create a class for the contract
class Contract:
  """ Base class for the contract. """
  def __init__(self, contract_type : str, amount : int, cost : float, 
  limitations : str, mv_link : str, contract_id : str,
  artist_address : str = '', counterparty_address : str = ''):
      """ Initialize the contract. """
      self.contract_type = contract_type
      self.amount = amount
      self.cost = cost
      self.limitations = limitations
      self.initiated = False
      self.artist_signed = False
      self.counterparty_signed = False
      self.total_cost = self.cost * self.amount
      self.developer_fee = self.calculate_fee()
      self.mv_link = mv_link
      self.contract_id = contract_id
      self.artist_address = artist_address
      self.counterparty_address = counterparty_address
      self.profit = self.total_cost - self.developer_fee

  def initiate_contract(self):
    """ Initiate the contract. """
    self.initiated = True

  def artist_sign_contract(self):
      """ Artist signs the contract. """
      self.artist_signed = True
  
  def counterparty_sign_contract(self):
      """ Counterparty signs the contract. """
      self.counterparty_signed = True
  
  def get_contract_status(self):
      """ Get the contract status. """
      return self.initiated, self.artist_signed, self.counterparty_signed
  
  def get_contract_details(self):
      """ Get the contract details. """
      return self.contract_type, self.amount, self.cost, self.limitations
  
  def get_contract(self):
      """ Get the contract. """
      return self
  
  def calculate_fee(self):
      """ Calculate the developer fee. """
      return self.total_cost * 0.05

# Connect to Ethereum (Replace with your provider URL)
#w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
#st.session_state.w3 = w3
# Contract ABI (Replace with the actual ABI of your contract
# Read in the contract ABI
#with open('./contracts/compiled/mv_contracts.json', 'r') as f:
#    contract_abi = f.read()

# Contract Address (Replace with the actual address of your deployed contract)
#contract_address = F'{os.getenv("CONTRACT_ADDRESS")}'

# Create Contract Instance
#contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
        'nft_demo_page', 'token_id', 'artist_signed', 'nft_metadata', 'label_wallet', 'receipt', 'label_signed', 'contract'
    ]
    default_values = [
        'nft_demo_home', 0, False, {}, '', {}, False, None
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_variables()

    
# Create a list of dictionaries for the contract types
contract_types = [
    {'type': 'MV Only', 'amount': 1, 'cost': 500, 'limitations': 'Use must be approved before deployment'},
    {'type': 'Co-writer', 'amount': 1, 'cost': 50, 'limitations': 'CW only'},
    {'type': 'Personalized Content', 'amount': 1, 'cost': 10, 'limitations': 'None'},
    {'type': 'Studio All Access', 'amount': 1, 'cost': 100000, 'limitations': 'None'}
]
        

#def get_metadata(token_id):
#    """ Fetches the metadata of the minted NFT. """
#    # Example: Fetching the metadata of the minted NFT (Replace with actual fetching)
#    metadata = contract.functions.getMetadata(token_id).call({'from': os.getenv('ARTIST_WALLET')})
#    return metadata

    
def minting_demo_home():
    """ Home page for the NFT minting demo. Allows the user
    to emulate the process of minting the NFT as an artist would."""
    # Introduction to NFT minting
    st.markdown("""
                <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
                The Artist Vault Protocol.  Hybrid blockchain security for transparency, accountability, and peace of mind.</p>
                """, unsafe_allow_html=True)
    st.markdown("""**By leveraging the advantages of both issuing NFT contracts on the public blockchain and storing sensitive information
    and financial transactions on a privately managed blockchain, we aim to make the process of onboarding artists onto the platform as intuitive
    and familiar as possible.  Even though the web3 ecosystem has become more and more mainstream over the past few years, it has by no means reached
    widespread adoption among a large swath of the population, particularly among a demographic that would include many of the artists we
    are targeting for our platform.  By layering on a privately managed blockchain that can handle fiat transactions and more traditional
    security measures for ease of access, we can alleviate the potential pain point of trying to set artists up with a crypto wallet,
    explain cryptocurencies, and introduce large amounts of friction that may prove problematic for many of our users.  There are also, of course,
    many issues with the web3 ecosystem, both fundamentally and reputationally.  One needs to look no further than the recent prosecution of Sam
    Bankman-Fried in one of the largest scams in American history involving his cryptocurrency exchange, FTX, to see why leaning in to
    web3 when approaching artists about securing their vocal legacy may prove problematic.  So with all that in mind, in the next few pages we will
    walk through the process of onboarding an artist via the secure Artist Vault protocol as well as issuing publicly verifiable NFT contracts representing
    the users consent and compensation for various downstream use cases.**""")

    # Step 1: Artist generates the NFT
    st.markdown("""
                <p style="font-family: 'Montserrat', sans-serif; color: #EDC480; font-size: 25px; font-weight: 550;">
    Step 1: Artist Generates the NFT
                </p>
    """, unsafe_allow_html=True)
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
    st.markdown("""
    <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
    font-size: 26px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Artist Actions</h4>
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 16px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">While the actual process of issuing NFTs
    and signing contracts on our private blockchain is obviously more complex, this will give you
    some sense of the process and how both the artist and First Rule can monetizes myriad downstream
    use cases.
</h3>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    # We will create a form to take in all of the relevant information
    # for minting the NFT
    # Load the artist wallet
    artist_wallet = os.getenv("ARTIST_WALLET")
    type_of_contract = st.selectbox('##### Type of Contract', options =
                                    ['MV Only', 'Co-writer', 'Personalized Content', 'Studio All Access'])
    # Depending on which type of contract is selected, we will display the relevant information
    # Set the cost of the contract
    if type_of_contract == 'MV Only':
        cost = 500
        limitations = 'Use must be approved before deployment'
    elif type_of_contract == 'Co-writer':
        cost = 50
        limitations = 'CW only'
    elif type_of_contract == 'Personalized Content':
        limitations = 'Approval Only'
        cost = 10
    else:
        limitations = 'None'
        cost = 100000
    # Set the amount of the contract
    contract_cost = st.number_input("##### Enter the amount of the contract:", value=cost, step=5)
    approved_artist_address = artist_wallet
    artist_address = artist_wallet
    #st.markdown("**:blue[@Joel - Below is where the artist's key would activate the minting process.]**")
    #artist_private_key = st.text_input("Enter your private key:", type='password', value = os.getenv("ARTIST_PRIVATE_KEY"))
    # Allow the user to enter the location of the relevant MV
    voice_print_choice = st.text_input("##### Enter the link to the MV:", type="password", value=f"https://mvstorage.com/{artist_address}")
    voice_print_link = f"https://mvstorage.com/{artist_address}" if not voice_print_choice else voice_print_choice
    # Enter the name of the counterparty
    label_address = st.text_input("##### Name of the counterparty:", value="Universal Music Group")
    # Set the contract_id
    contract_id = str(uuid.uuid4()) # Generate a random UUID
    # Get the limitations of the contract
    # Create a button to mint the NFT
    # Load in the logo
    #logo = Image.open("./resources/vl_logo1.png")
    #st.image(logo, width=100)
    st.text("")
    mint_nft_button = st.button("Sign Contract and Mint NFT", type='primary', use_container_width=True)
   
    if mint_nft_button:
        with st.spinner("Minting NFT..."):
            time.sleep(3)
            # Validate the form inputs
            if artist_address != approved_artist_address:
                st.error("You are not authorized to mint this NFT or the label address is\
                        not approved. Please select an approved wallet address.")    
            # Create a new NFT contract and set the session state
            st.session_state.contract = Contract(type_of_contract, 1, contract_cost, limitations, 
                                                        voice_print_link, contract_id, artist_address, label_address)
           
            # Call the initiateNFT function from the contract (Replace with correct function name and parameters)
            #tx = contract.functions.initiateNFT(value, contract_verbiage, reason, voice_print_link, label_address).build_transaction(
            #    {
            #        'from': artist_wallet,
            #        'gas': 1000000,
            #        'nonce': w3.eth.get_transaction_count(artist_wallet),
            #        'gasPrice': w3.eth.gas_price  # Manually set the gas price
            #    }
            #)
            #signed_tx = w3.eth.account.sign_transaction(tx, private_key=os.getenv("ARTIST_KEY"))
            #tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            #receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            #rich_logs = contract.events.NFTInitiated().process_receipt(receipt)
            #token_id = rich_logs[0]['args']['tokenId']
            #st.session_state.latest_nft = token_id
            #st.session_state.artist_signed = True
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
    st.text("")
    st.text("")
    # Display available NFTs
    st.markdown("""
    <div style='display: block; width: 100%;'>
    <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #fff;
    font-size: 18px; font-weight: 550;  animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
    fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">
    Below we display a sample contract that a counterparty would see when
    they are signing an NFT.  The contract is generated by the artist and
    is immutable once it is signed by both parties.  The counterparty can
    view the contract and sign it if they agree to the terms.  Once both
    parties have signed, payment will be handled securely behind our private blockchain, and the
    the NFT representing the contract is minted on the Ethereum blockchain for verification and transparency.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Display the details of the NFT
    st.markdown(f"##### Contract ID: {st.session_state.contract.contract_id}")
    st.markdown(f"##### Contract Type: {st.session_state.contract.contract_type}")
    st.markdown(f"##### Cost: ${st.session_state.contract.total_cost:.2f}")
    st.markdown(f"##### Limitations: {st.session_state.contract.limitations}")

    # Allow the user to select their wallet address
    wallet_address = label_wallet
    # User inputs their private key
    #private_key = st.text_input("Enter your private key:", type="password", value = os.getenv("LABEL_KEY"))
    #private_key = os.getenv("LABEL_KEY")
    approved_wallet_address = label_wallet
    st.text("")
    # Signing process
    if st.warning("Once you have confirmed the details of the contract, sign by clicking the button below\
                   to receive the full NFT metadata."):
        # Create a button to sign the contract
        sign_button = st.button("Sign Contract", type='primary', use_container_width=True)
        if sign_button:
            if wallet_address != approved_wallet_address:
                st.error("You are not authorized to sign this contract. Please select an approved wallet address.")
                st.stop()
            ## Call the signNFT function from the contract
            #tx = contract.functions.counterpartySign(st.session_state.latest_nft).build_transaction(
            #    {
            #        "from": label_wallet,
            #        'gas': 1000000,
            #        'nonce': w3.eth.get_transaction_count(label_wallet),
            #        'gasPrice': w3.eth.gas_price  # Manually set the gas price
            #    }
            #)
            #signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
            #tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            #receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            #logs = contract.events.NFTSigned().process_receipt(receipt)
            #st.session_state.label_signed = True
             
            # Display a success message with the transaction hash
            #if st.session_state.artist_signed:
            #    st.success(f"Successfully signed NFT # {st.session_state.latest_nft}!\
            #    \nTransaction Hash: {tx_hash.hex()}")
            #else:
            #    st.success(f"Successfully signed NFT # {st.session_state.latest_nft}!\
            #    \nTransaction Hash: {tx_hash.hex()}\
            #    \nWaiting for artist to sign...")
        
        # Display a message if the user has already signed
        #if st.session_state.label_signed:
            # Create a button to view the NFT
            #view_nft_button = st.button("View NFT", type='primary', use_container_width=True)
            #if view_nft_button:
            st.session_state.nft_demo_page = "display_nft_metadata"
            st.experimental_rerun()

def display_nft_metadata():
    """ Displays the metadata of the NFT."""
    st.markdown("""
                <p style="font-family: 'Montserrat', sans-serif; color: #fff; font-size: 20px; font-weight: 550;">
                Congratulations! The artist has approved your contract and you can begin creating magic with their
                digital likeness by your side.</p>
                """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Contract ID:** {st.session_state.contract.contract_id}")
    st.markdown(f"**Contract Type:** {st.session_state.contract.contract_type}")
    st.markdown(f"**Total Cost**: ${st.session_state.contract.total_cost:.2f}")
    st.markdown(f"**Developer Fee**: ${st.session_state.contract.developer_fee:.2f}")
    st.markdown(f"**Artist Profit**: ${st.session_state.contract.profit:.2f}")
    st.markdown(f"**Limitations:** {st.session_state.contract.limitations}")
    st.markdown(f"**MV Link:** {st.session_state.contract.mv_link}")

    # Create a button to move on to the Artist Chat Page
    st.markdown('---')
    st.markdown('**Creating and licensing melodic voiceprints as NFTs** offers scalable revenue opportunities\
                 for artists while enhancing protection.  We are particularly excited about one such possibility,\
                code-named **"Co-writer"**, that will revolutionize how artists can interact with their fans, fellow musicians,\
                studios, and others. By training unique large language models with the artist\'s approved data,\
                this introduces a groundbreaking level of personalization, shifting the way music is created and experienced.'
                )
    chat_with_artist_button = st.button("Explore Co-writer", type='primary', use_container_width=True)
    if chat_with_artist_button:
        st.session_state.contract = None
        st.session_state.nft_demo_page = "artist_minting_demo"
        switch_page("Co-writer")
    mint_new_nft_button = st.button("Mint a new NFT", type='primary', use_container_width=True)
    if mint_new_nft_button:
        st.session_state.contract = None
        st.session_state.nft_demo_page = "artist_minting_demo"
        st.experimental_rerun()
    back_to_home_button = st.button("Back to Home", type='primary', use_container_width=True)
    if back_to_home_button:
        st.session_state.contract = None
        st.session_state.nft_demo_page = "artist_minting_demo"
        switch_page("main")
    
# Call the function to display the page
if st.session_state.nft_demo_page == "nft_demo_home":
    artist_minting_demo()
elif st.session_state.nft_demo_page == "artist_minting_demo":
    artist_minting_demo()
elif st.session_state.nft_demo_page == "label_signing_demo":
    label_interface()
elif st.session_state.nft_demo_page == "display_nft_metadata":
    display_nft_metadata()
