// SPDX-License-Identifier: MIT
pragma solidity ^0.8.6;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract VoiceprintNFT is ERC721URIStorage, Ownable {
    uint256 public tokenCounter;
    uint256 public developerPercentage;
    address payable developerAddress;

    struct VoicePrint {
        string voicePrintReference;
        uint256 salePrice;
    }

    mapping (uint256 => VoicePrint) public voiceprints;

    event Mint(address to, uint256 tokenId);

    constructor (address payable _developerAddress) ERC721("VoiceprintNFT", "VPRINT") {
        tokenCounter = 0;
        developerPercentage = 5; // 5 percent cut to developer
        developerAddress = _developerAddress;
    }

    function mintNFT(address recipient, string memory tokenURI, string memory voicePrintReference, uint256 salePrice) public onlyOwner {
        voiceprints[tokenCounter] = VoicePrint(voicePrintReference, salePrice);
        _mint(recipient, tokenCounter);
        _setTokenURI(tokenCounter, tokenURI);

        emit Mint(recipient, tokenCounter);

        tokenCounter++;
    }

    function buyNFT(uint256 tokenId) public payable {
        VoicePrint memory voiceprint = voiceprints[tokenId];
        require(msg.value >= voiceprint.salePrice, "Insufficient funds sent");

        uint256 developerCut = (msg.value * developerPercentage) / 100;
        developerAddress.transfer(developerCut);

        uint256 sellerCut = msg.value - developerCut;
        address payable sellerAddress = payable(ownerOf(tokenId));
        sellerAddress.transfer(sellerCut);

        _transfer(ownerOf(tokenId), msg.sender, tokenId);
    }
}
