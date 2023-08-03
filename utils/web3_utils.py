""" Utilities related to interacting with the Ethereum blockchain.
This will utilize the web3.py library and reference a Vyper smart contract
named VoiceprintNFT"""
from web3 import Web3
from eth_account.messages import encode_defunct
from eth_account import Account, messages
from eth_utils import keccak
from eth_typing import Address
import requests
import streamlit as st
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Load the environment variables
crypto_key = os.getenv("CRYPTO_API_KEY")

def convert_to_ether(usd_amount: float) -> float:
    """ Convert an amount of dollars to Ether.]
    We will use a fixed conversion rate """
    url = "https://min-api.cryptocompare.com/data/price"
    # Get the current price of Ether in USDC
    eth_price = requests.get(url=url, params={'fsym': 'ETH', 'tsyms': 'USDC'}, timeout=10).json()['USDC']
    # Convert the amount to Ether
    ether_amount = usd_amount / eth_price
    return ether_amount

def convert_to_wei(amount_ether: float) -> int:
    """ Convert a float amount of Ether to Wei."""
    return Web3.toWei(amount_ether, 'ether')

def mint_nft(recipient: Address, voicePrintReference: str, _salePrice: int):
    """ Mint a new NFT and transfer it to the recipient address."""
    return {"recipient": recipient, "voicePrintReference": voicePrintReference, "salePrice": _salePrice}