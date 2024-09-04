import requests
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from decimal import Decimal
from web3 import Web3
import datetime
import time 
from dataclass_csv import DataclassReader, DataclassWriter
from typing import Optional
from web3.types import HexBytes
from data import AssetTransfer, SwapEvent, Log, load_from_csv, save_to_csv
from utils import get_block_by_timestamp
import pytz 

chain = "polygon"
url = f"https://polygon-mainnet.g.alchemy.com/v2/{os.environ.get('ALCHEMY_API_KEY')}"
w3 = Web3(Web3.HTTPProvider(url))
uniswap_router_abi = json.load(open("abis/uniswap_router.json"))

# Read constants.json
with open('constants.json', 'r') as f:
    constants = json.load(f)


from_block = get_block_by_timestamp(datetime.datetime(2023, 9, 1, 0, 0, 0, tzinfo=pytz.utc), chain)
switch_block = get_block_by_timestamp(datetime.datetime(2023, 10, 17, 0, 0, 0, tzinfo=pytz.utc), chain)
to_block = get_block_by_timestamp(datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc), chain)


print(from_block, switch_block, to_block)

def get_asset_transfers(address, from_block, to_block):
    payload = {
        "id": 1,
    "jsonrpc": "2.0",
    "method": "alchemy_getAssetTransfers",
    "params": [
        {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "toAddress": address,
            "order": "asc",
            "withMetadata": True,
            "excludeZeroValue": True,
            "maxCount": "0x3e8",
            "category": ["erc20"]
        }
    ]
}
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())
    return list(map(AssetTransfer.from_dict, response.json()["result"]["transfers"]))

def get_swaps(address, from_block, to_block):
    contract = w3.eth.contract(address=address, abi=uniswap_router_abi)
    logs = contract.events.Swap().get_logs(from_block=from_block, to_block=to_block)
    return list(map(SwapEvent.from_dict, logs))

def get_asset_transfer_pool(router_address: str, hash: str):
    # Read the logs and determine which pool does the swap take place in 
    receipt = w3.eth.get_transaction_receipt(hash)
    for _log in receipt.logs:
        log = Log.from_dict(_log)
        padded_router_address = '0x' + router_address[2:].lower().zfill(64)
        if log.topics[0] == constants["TRANSFER_TOPIC"] and log.topics[2] == padded_router_address:
            return "0x" + log.topics[1][26:]
    return None

ALCHEMY_MAX_LOG_WINDOW = 2000
def alchemy_call_batch_blocks(func, cls, task_name, from_block, to_block, *args):
    logs = []
    filename = f"{task_name}_{from_block}_{to_block}.csv"
    if os.path.exists(filename):
        logs = load_from_csv(filename, cls)
        from_block = int(logs[-1].blockNum, 16) + 1 if logs[-1].blockNum.startswith('0x') else int(logs[-1].blockNum) + 1

    current_from_block = from_block
    current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW, to_block)
    while current_from_block < to_block:
        try:
            logs.extend(func(*args, current_from_block, current_to_block))
            save_to_csv(logs, filename, cls)
            current_from_block = current_to_block 
            current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW , to_block)
        except Exception as e:
            print(e, "retrying...")
            time.sleep(3)

        time.sleep(1)   
    return logs

def split_asset_transfers_by_pool(asset_transfers: list[AssetTransfer], from_block, to_block):

    # Pre-processing
    pool_transfers = dict()
    for i, transfer in enumerate(asset_transfers):
        pool = get_asset_transfer_pool(constants["POLYGON_UNISWAP_ROUTER_V3"], transfer.hash).lower()  
        pool_transfers[pool] = pool_transfers.get(pool, []) + [transfer]
        if i % 100 == 0 or i == len(asset_transfers) - 1:
            for pool, transfers in pool_transfers.items():
                save_to_csv(transfers, f"asset_transfers_by_pool/{pool}_{from_block}_{to_block}.csv", AssetTransfer)


alchemy_call_batch_blocks(get_asset_transfers,  AssetTransfer, "asset_transfers", 48810868, 50600174, constants["POLYGON_UNISWAP_FEECOLLECTOR_LEGACY"])
alchemy_call_batch_blocks(get_swaps, SwapEvent, "swaps", 48810868, 50600174, constants["POLYGON_UNISWAP_SWAP_CONTRACT"])
