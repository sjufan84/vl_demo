""" Utilities related to interacting with the Ethereum blockchain.
This will utilize the web3.py library and reference a Vyper smart contract
named VoiceprintNFT"""
import os
import requests


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
    return amount_ether * 10**18

