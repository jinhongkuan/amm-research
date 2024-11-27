import requests
import datetime
import time
from web3 import Web3
import pandas as pd
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

def get_block_gas_price(w3: Web3, block: int) -> int:
    blockData =  w3.eth.get_block(block)
    gas_price = blockData.get("baseFeePerGas")
    if gas_price is None: 
        raise Exception(f"Block {block} not found")
    return gas_price

if __name__ == "__main__":
    # Initialize empty list to store data
    data = []
    w3 = Web3(Web3.HTTPProvider("https://mainnet.gateway.tenderly.co"))
    
    try:
        for block in range(18039180, 18689339):
            gas_price = get_block_gas_price(w3, block)
            data.append({"block": block, "gas_price": gas_price})
            
        # Create DataFrame from collected data
        df = pd.DataFrame(data)
        df.to_csv("gas_prices.csv", index=False)
    except Exception as e:
        print(f"Error occurred: {e}")

