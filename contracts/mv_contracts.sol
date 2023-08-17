// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

// Contract to represent the NFT
contract ArtistNFT is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

// Events
    event NFTInitiated(uint256 indexed tokenId, address indexed artist, address indexed counterparty);
    event NFTSigned(uint256 indexed tokenId, address indexed counterparty, address indexed artist);
    event NFTMinted(uint256 indexed tokenId, address artist, address indexed counterparty);
    event CommissionSent(uint256 indexed tokenId, uint256 indexed amount, address indexed developer);

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
    address payable public owner; // Owner of the contract
    address payable private developerAddress; // Developer's address for commission
    uint256 private constant COMMISSION_PERCENT = 5; // 5% commission

    modifier onlyOwner {
    require(msg.sender == owner, "Ownable: You are not the owner, Bye.");
    _;
  }

    // Constructor -- sets the developer's address
    constructor(address payable _developerAddress) ERC721("ART", "ART") {
        developerAddress = _developerAddress;
        owner = msg.sender;
    }

    // Primary function to initiate the NFT
    function initiateNFT(
            uint256 value,
            string memory contractVerbiage,
            string memory voicePrintLink,
            string memory reason, 
            address payable memory counterparty
        ) public onlyOwner returns (uint256) { // Only owner can initiate the NFT
            _tokenIds.increment(); // Increment token ID
            uint256 newItemId = _tokenIds.current(); // Get current token ID
            _metadata[newItemId] = NFTMetadata(value, contractVerbiage, voicePrintLink, reason, true, false, counterparty); // Set metadata
            
            emit NFTInitiated(newItemId, msg.sender, counterparty); // Emit event
            return newItemId;
        }

    function counterpartySign(uint256 tokenId) public {
        // require that the caller is the appropriate counterparty
        require(msg.sender == _metadata[tokenId]._counterparty, "Only counterparty can sign");
        require(!_metadata[tokenId]._counterpartySigned, "Already signed by counterparty");
        _metadata[tokenId]._counterpartySigned = true;

        emit NFTSigned(tokenId, msg.sender, this.owner); // Emit event
        
        if (_metadata[tokenId]._artistSigned) {
            mintNFT(tokenId);
        }
    }

    function mintNFT(uint256 tokenId) internal {
        require(_metadata[tokenId].artistSigned && _metadata[tokenId].counterpartySigned, "Both parties must sign");
        _mint(owner(), tokenId);
        _setTokenURI(tokenId, ""); // Set token URI

        emit NFTMinted(tokenId, owner()); // Emit event

        // Send commission to developer
        uint256 commission = (_metadata[tokenId].value * COMMISSION_PERCENT) / 100;
        developerAddress.transfer(commission);

        emit CommissionSent(tokenId, commission, developerAddress); // Emit event
    }
    
    function getMetadata(uint256 tokenId) public view returns (NFTMetadata memory) {
        return _metadata[tokenId];
    }

    function setDeveloperAddress(address payable _developerAddress) public onlyOwner {
        require(_developerAddress != address(0), "Invalid address");
        developerAddress = _developerAddress;
    }

    function getDeveloperAddress() public view returns (address) {
        return developerAddress;
    }
}

