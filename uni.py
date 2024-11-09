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
from data import AssetTransfer, SwapEvent, SwapEventV2, Log, load_from_csv, save_to_csv, MintEvent, BurnEvent
from utils import get_block_by_datetime
import uniswap
import pytz 
import csv 

chain = "arbitrum"
url = f"https://arbitrum.gateway.tenderly.co"
w3 = Web3(Web3.HTTPProvider(url))
uniswap_router_abi = json.load(open("abis/uniswap_router.json"))
uniswap_v2_pool_abi = json.load(open("abis/uniswap_v2_pool.json"))
uni = uniswap.Uniswap(address=None, private_key=None,web3=w3, version=3)

# Read constants.json
with open('constants.json', 'r') as f:
    constants = json.load(f)

# from_time = datetime.datetime(2023, 9, 1, 0, 0, 0, tzinfo=pytz.utc)
# switch_time = datetime.datetime(2023, 10, 17, 0, 0, 0, tzinfo=pytz.utc)
# to_time = datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc)

# from_block = get_block_by_datetime(from_time, chain)
# switch_block = get_block_by_datetime(switch_time, chain)
# to_block = get_block_by_datetime(to_time, chain)



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

def get_swaps(chain, address, from_block, to_block):
    w3 = Web3(Web3.HTTPProvider(constants[chain]["rpc_url"]))
    contract = w3.eth.contract(address=address, abi=uniswap_router_abi)
    logs = contract.events.Swap().get_logs(fromBlock=from_block, toBlock=to_block)
    return list(map(SwapEvent.from_dict, logs))

def get_mints(address, from_block, to_block):
    contract = w3.eth.contract(address=address, abi=uniswap_v2_pool_abi)
    logs = contract.events.Mint().get_logs(from_block=from_block, to_block=to_block)
    return list(map(MintEvent.from_dict, logs))

def get_burns(address, from_block, to_block):
    contract = w3.eth.contract(address=address, abi=uniswap_v2_pool_abi)
    logs = contract.events.Burn().get_logs(from_block=from_block, to_block=to_block)
    return list(map(BurnEvent.from_dict, logs))

def get_v2_pool_swaps(address, from_block, to_block):
    contract = w3.eth.contract(address=address, abi=uniswap_v2_pool_abi)
    logs = contract.events.Swap().get_logs(from_block=from_block, to_block=to_block)
    return list(map(SwapEventV2.from_dict, logs))


ALCHEMY_MAX_LOG_WINDOW = 1000
FILE_CHUNK_ENTRIES = 1000000 
def alchemy_call_batch_blocks(func, cls, task_name, from_block, to_block, *args):
    logs = []
    current_from_block = from_block
    current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW, to_block)
    file_index = 0

    while current_from_block < to_block:
        filename = f"{task_name}_{current_from_block}_{min(current_from_block + FILE_CHUNK_ENTRIES - 1, to_block)}.csv"
        
        if os.path.exists(filename):
             # Determine if the range has been completed 
            next_file_index = file_index + 1
            next_file_from_block = from_block + next_file_index * FILE_CHUNK_ENTRIES
            next_file_to_block = min(next_file_from_block + FILE_CHUNK_ENTRIES - 1, to_block)
            if os.path.exists(f"{task_name}_{next_file_from_block}_{next_file_to_block}.csv"):
                current_from_block = next_file_to_block + 1
                current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW, to_block)
                file_index += 1
                print("skipping", current_from_block, current_to_block)
                continue

            chunk_logs = load_from_csv(filename, cls)
            if chunk_logs:
                logs.extend(chunk_logs)
                if cls == AssetTransfer:
                    current_from_block = int(chunk_logs[-1].blockNum, 16) + 1 if chunk_logs[-1].blockNum.startswith('0x') else int(chunk_logs[-1].blockNum) + 1
                elif cls in [SwapEvent, SwapEventV2]:
                    current_from_block = chunk_logs[-1].blockNumber + 1
                else:
                    raise ValueError(f"Unsupported class: {cls}")
                current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW, to_block)
                continue
           

        chunk_logs = []
        while current_from_block < min(from_block + (file_index + 1) * FILE_CHUNK_ENTRIES, to_block):
            try:
                chunk_logs.extend(func(*args, current_from_block, current_to_block))
                current_from_block = current_to_block
                current_to_block = min(current_from_block + ALCHEMY_MAX_LOG_WINDOW, to_block)
            except Exception as e:
                print(f"Error at block {current_from_block}: {e}, retrying...")
                time.sleep(3)
                continue

            time.sleep(1)
        logs.extend(chunk_logs)
        save_to_csv(chunk_logs, filename, cls)
        file_index += 1

    return logs

def combine_csv_files(task_name, cls, from_block, to_block):
    logs = []
    while from_block < to_block:
        filename = f"{task_name}_{from_block}_{min(from_block + FILE_CHUNK_ENTRIES - 1, to_block)}.csv"
        print(f"Loading {filename}")
        if not os.path.exists(filename):
            break
        chunk_logs = load_from_csv(filename, cls)  
        logs.extend(chunk_logs)
        from_block = min(from_block + FILE_CHUNK_ENTRIES, to_block)
    return logs 
        
def get_v2_total_supply(address, from_block, to_block):
    contract = w3.eth.contract(address=address, abi=uniswap_v2_pool_abi)
    total_supply = []
    for i in range(from_block, to_block):
        try:
            total_supply.append(contract.functions.totalSupply().call(block_identifier=i))
            if i % 100 == 0 or i == to_block - 1:
                json.dump(total_supply, open(f"total_supply_{from_block}_{to_block}.json", "w"))
        except Exception as e:
            print(e, "retrying...")
            time.sleep(3)
    return total_supply

# alchemy_call_batch_blocks(get_v2_pool_swaps,  SwapEventV2, "swaps_ethereum_v2_300",  from_block, to_block, constants["ethereum"]["v2_300"])
# alchemy_call_batch_blocks(get_swaps,  SwapEvent, "swaps_arbitrum_v3_50",  from_block, to_block, constants["arbitrum"]["v3_50"])
# total_supply_column = get_v2_total_supply(constants["ethereum"]["v2_300"], from_block, to_block) 3.126285639228751e17

# pool = uni.get_pool_instance(constants["arbitrum"]["tokens"]["USDC"], constants["arbitrum"]["tokens"]["USDT"], 500)
