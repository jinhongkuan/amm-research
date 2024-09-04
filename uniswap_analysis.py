from data import load_from_csv, save_to_csv, SwapEvent, AssetTransfer
from utils import get_block_by_timestamp
import datetime
from uniswap import get_asset_transfer_pool
import json

chain = "polygon"
from_block = get_block_by_timestamp(datetime.datetime(2023, 11, 1, 0, 0, 0), chain)
to_block = from_block + 20000

# Read constants.json
with open('constants.json', 'r') as f:
    constants = json.load(f)
pool = constants["POLYGON_UNISWAP_SWAP_CONTRACT"]
swaps : list[SwapEvent] = load_from_csv(f"swaps_{from_block}_{to_block}.csv", SwapEvent)
asset_transfers : list[AssetTransfer] = load_from_csv(f"asset_transfers_by_pool/{pool.lower()}_{from_block}_{to_block}.csv", AssetTransfer)

# Separate swap into two categories:
# 1. Swap that does not have a matching asset transfer
# 2. Swap that has a matching asset transfer - this implies frontend fees are incurred

def match_swap_asset_transfer(swap: SwapEvent, asset_transfer: AssetTransfer) -> bool:
    return swap.transactionHash.lower() == asset_transfer.hash.lower()

def scan_swaps_for_fees(swaps: list[SwapEvent], asset_transfers: list[AssetTransfer]):
    swaps_no_fees : list[SwapEvent] = []
    swaps_with_fees : list[(SwapEvent, AssetTransfer)] = []

    swap_cursor = 0
    for transfer in asset_transfers:
        block_number = int(transfer.blockNum, 16) if transfer.blockNum.startswith('0x') else int(transfer.blockNum)
        block_matched = False
        swap_matched = False
        for i, swap in enumerate(swaps[swap_cursor:]):
            if swap.blockNumber == block_number and not block_matched:
                block_matched = True
                swap_cursor = i

            if match_swap_asset_transfer(swap, transfer):
                swaps_with_fees.append((swap, transfer))
                swap_matched = True
                break
      

        if not swap_matched:
            raise Exception(f"Swap for {transfer.hash} not found")

    # Any swaps that are not matched with a transfer belong to swaps_no_fees
    swaps_no_fees = [swap for swap in swaps if swap.transactionHash not in [s.transactionHash for s, _ in swaps_with_fees]]

    return swaps_no_fees, swaps_with_fees

swaps_no_fees, swaps_with_fees = scan_swaps_for_fees(swaps, asset_transfers)
print(swaps_with_fees[0])