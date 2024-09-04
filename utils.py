import requests
import datetime
import time

def get_block_by_timestamp(timestamp: datetime.datetime, chain: str) -> int:
    url = f"https://coins.llama.fi/block/{chain}/{int(timestamp.timestamp())}"
    response = requests.get(url)
    return int(response.json()["height"])
