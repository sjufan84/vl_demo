// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

contract ArtistNFT is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

// Events
    event NFTInitiated(uint256 tokenId, address artist);
    event NFTSigned(uint256 tokenId, address signer);
    event NFTMinted(uint256 tokenId, address artist);
    event CommissionSent(uint256 tokenId, uint256 amount, address developer);

    // Structure to represent the NFT metadata
    struct NFTMetadata {
        uint256 value;
        string contractVerbiage;
        string voicePrintLink;
        string reason;
        bool artistSigned;
        bool counterpartySigned;
    }

    mapping(uint256 => NFTMetadata) private _metadata;
    address payable private developerAddress; // Developer's address for commission
    uint256 private constant COMMISSION_PERCENT = 5; // 5% commission

    constructor(address payable _developerAddress) ERC721("ART", "ART") {
        developerAddress = _developerAddress;
    }

    // Rest of the minting and signature logic will be implemented in subsequent steps

    function initiateNFT(
            uint256 value,
            string memory contractVerbiage,
            string memory voicePrintLink,
            string memory reason
        ) public onlyOwner returns (uint256) {
            _tokenIds.increment();
            uint256 newItemId = _tokenIds.current();
            _metadata[newItemId] = NFTMetadata(value, contractVerbiage, voicePrintLink, reason, true, false);
            
            emit NFTInitiated(newItemId, msg.sender); // Emit event
            return newItemId;
        }

    function counterpartySign(uint256 tokenId) public {
        require(!_metadata[tokenId].counterpartySigned, "Already signed by counterparty");
        _metadata[tokenId].counterpartySigned = true;

        emit NFTSigned(tokenId, msg.sender); // Emit event
        
        if (_metadata[tokenId].artistSigned) {
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

