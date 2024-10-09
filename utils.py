import requests
import datetime
import time
from web3 import Web3

def get_block_by_datetime(timestamp: datetime.datetime, chain: str) -> int:
    url = f"https://coins.llama.fi/block/{chain}/{int(timestamp.timestamp())}"
    response = requests.get(url)
    return int(response.json()["height"])

def get_block_timestamp(w3: Web3, block: int) -> int:
    blockData =  w3.eth.get_block(block)
    timestamp = blockData.get("timestamp")
    if timestamp is None: 
        raise Exception(f"Block {block} not found")
    return timestamp
