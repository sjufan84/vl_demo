// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// Contract to represent the NFT
contract ArtistNFT is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

    // Define roles
    // bytes32 public constant ARTIST_ROLE = keccak256("ARTIST_ROLE");

    // Define developer address
    address payable _developerAddress;

    // Events
    event NFTInitiated(uint256 indexed tokenId, address indexed artist, address indexed counterparty);
    event NFTSigned(uint256 indexed tokenId, address indexed counterparty, address indexed artist);
    event NFTMinted(uint256 indexed tokenId, address indexed artist, address indexed counterparty);

    // Structure to represent the NFT metadata
    struct NFTMetadata {
        uint256 _value;
        string _contractVerbiage;
        string _voicePrintLink;
        string _reason;
        bool _artistSigned;
        bool _counterpartySigned;
        address payable _counterparty;
    }

    // Mapping to store the NFT metadata
    mapping(uint256 => NFTMetadata) private _metadata;

    // Constructor
    constructor(address payable developerAddress) ERC721("ART", "ART") {
        _developerAddress = developerAddress; // Set the developer address
    }

    // Primary function to initiate the NFT
    function initiateNFT(
        uint256 value,
        string memory contractVerbiage,
        string memory voicePrintLink,
        string memory reason, 
        address payable counterparty
    ) public returns (uint256) {
        _tokenIds.increment(); // Increment token ID
        uint256 newItemId = _tokenIds.current(); // Get current token ID
        _metadata[newItemId] = NFTMetadata(value, contractVerbiage, voicePrintLink, reason, true, false, counterparty); // Set metadata

        emit NFTInitiated(newItemId, msg.sender, counterparty); // Emit event
        return newItemId;
    }

    // Function to sign as counterparty
    function counterpartySign(uint256 tokenId) public {
        // require that the caller is the appropriate counterparty
        require(msg.sender == _metadata[tokenId]._counterparty, "Only counterparty can sign");
        require(!_metadata[tokenId]._counterpartySigned, "Already signed by counterparty");
        _metadata[tokenId]._counterpartySigned = true;

        emit NFTSigned(tokenId, msg.sender, _metadata[tokenId]._counterparty); // Emit event
        
        if (_metadata[tokenId]._artistSigned) {
            mintNFT(tokenId);
        }
    }

    // Function to mint NFT
    function mintNFT(uint256 tokenId) internal {
        require(_metadata[tokenId]._artistSigned && _metadata[tokenId]._counterpartySigned, "Both parties must sign");
        _mint(_metadata[tokenId]._counterparty, tokenId);
        string memory tokenURI = _constructTokenURI(_metadata[tokenId]._voicePrintLink); // Construct token URI
        _setTokenURI(tokenId, tokenURI); // Set token URI

        emit NFTMinted(tokenId, msg.sender, _metadata[tokenId]._counterparty); // Emit event
    }

    // Function to construct the token URI based on voicePrintLink
    function _constructTokenURI(string memory voicePrintLink) internal pure returns (string memory) {
        // You can construct the URI based on the voice print link or any other logic suitable for your application
        return string(abi.encodePacked("https://yourdomain.com/metadata/", voicePrintLink));
    }

    function getMetadata(uint256 tokenId) public view returns (NFTMetadata memory) {
        return _metadata[tokenId];
    }
}
    