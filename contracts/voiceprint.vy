""" Primary contract for Voiceprint NFTs
The contract will mint NFTs that are a representation
of a contract between an artist and another party.
This could be references to a model, a voiceprint, or
or some other form of digital asset.  Only the artist,
who is the owner of the contract, or an authorized party
should be able to mint the NFTs.  The sale should automatically
transfer 5% of the sale price to the developer's wallet.  The 
artist eill keep the rest.  The artist should be able to set the 
price of the NFTs.  There will be detailed metadata associated with
each NFT representing the verbiage of the agreement between the 
two parties.  The NFTs will be ERC721 compliant."""

# VoiceprintNFT.vy
from vyper.interfaces import ERC721

# Initialize the contract
owner: public(address)
developer: public(address)
artist: public(address)
price: public(uint256)
totalSupply: public(uint256)
contractURI: public(string[64])
tokenURI: public(string[64])

# Events
Transfer: event({_from: indexed(address), _to: indexed(address), _tokenId: uint256})
Approval: event({_owner: indexed(address), _approved: indexed(address), _tokenId: uint256})
ApprovalForAll: event({_owner: indexed(address), _operator: indexed(address), _approved: bool})
