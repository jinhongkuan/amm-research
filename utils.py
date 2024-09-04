import requests
import datetime
import time

def get_block_by_timestamp(timestamp: datetime.date, chain: str) -> int:
    url = f"https://coins.llama.fi/block/{chain}/{int(time.mktime(timestamp.timetuple()))}"
    response = requests.get(url)
    return int(response.json()["height"])
