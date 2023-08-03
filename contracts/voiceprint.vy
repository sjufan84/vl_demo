# VoiceprintNFT.vy
from vyper.interfaces import ERC721

# Interface for self, to be used by other contracts
interface VoiceprintNFT_Interface:
    def mintNFT(recipient: address, tokenURI: String[256], voicePrintReference: String[256], salePrice: uint256) -> uint256: modifying
    def buyNFT(tokenId: uint256): modifying

# ERC721 contract interface
interface ERC721_Interface:
    def ownerOf(tokenId: uint256) -> address: view
    def transferFrom(_from: address, _to: address, tokenId: uint256): modifying
    def setTokenURI(tokenId: uint256, tokenURI: String[256]): modifying

contract VoiceprintNFT:

    # Events
    event Mint:
        recipient: address
        tokenId: uint256

    # Storage
    tokenCounter: public(uint256)
    developerPercentage: public(uint256)
    developerAddress: public(address)

    voiceprints: public(map(uint256, VoicePrint))

    ERC721_contract: ERC721_Interface

    struct VoicePrint:
        voicePrintReference: String[256]
        salePrice: uint256

    # Initialize the contract
    @external
    def __init__(_developerAddress: address, ERC721_address: address):
        self.tokenCounter = 0
        self.developerPercentage = 5
        self.developerAddress = _developerAddress
        self.ERC721_contract = ERC721_Interface(ERC721_address)

    @external
    def mintNFT(_recipient: address, _tokenURI: String[256], _voicePrintReference: String[256], _salePrice: uint256) -> uint256:
        self.voiceprints[self.tokenCounter] = VoicePrint({voicePrintReference: _voicePrintReference, salePrice: _salePrice})
        self.ERC721_contract.setTokenURI(self.tokenCounter, _tokenURI)
        self.ERC721_contract.transferFrom(ZERO_ADDRESS, _recipient, self.tokenCounter)
        
        log Mint(_recipient, self.tokenCounter)

        self.tokenCounter += 1

    @external
    def buyNFT(_tokenId: uint256):
        voiceprint: VoicePrint = self.voiceprints[_tokenId]
        assert msg.value >= voiceprint.salePrice, "Insufficient funds sent"

        developerCut: uint256 = (msg.value * self.developerPercentage) / 100
        send(self.developerAddress, developerCut)

        sellerCut: uint256 = msg.value - developerCut
        sellerAddress: address = self.ERC721_contract.ownerOf(_tokenId)
        send(sellerAddress, sellerCut)

        self.ERC721_contract.transferFrom(self.ERC721_contract.ownerOf(_tokenId), msg.sender, _tokenId)
