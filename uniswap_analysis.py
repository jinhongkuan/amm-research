from data import load_from_csv, save_to_csv, SwapEvent, SwapEventV2, AssetTransfer
from utils import get_block_by_timestamp
import datetime
from uniswap import get_asset_transfer_pool
import json
import pandas as pd
import pytz
import statsmodels.formula.api as sm
import numpy as np

chain = "ethereum"
from_block = get_block_by_timestamp(datetime.datetime(2023, 9, 1, 0, 0, 0, tzinfo=pytz.utc), chain)
switch_block = get_block_by_timestamp(datetime.datetime(2023, 10, 17, 0, 0, 0, tzinfo=pytz.utc), chain)
to_block = get_block_by_timestamp(datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc), chain)


# Read constants.json
# with open('constants.json', 'r') as f:
#     constants = json.load(f)
# pool = constants["POLYGON_UNISWAP_SWAP_CONTRACT"]
# swaps : list[SwapEvent] = load_from_csv(f"swaps_{from_block}_{to_block}.csv", SwapEvent)
# asset_transfers : list[AssetTransfer] = load_from_csv(f"asset_transfers_by_pool/{pool.lower()}_{from_block}_{to_block}.csv", AssetTransfer)

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

def combine_swaps(v2_swaps: list[SwapEventV2], v3_swaps: list[SwapEvent], epoch_interval: int, treatment_block: int):
    columns = ["epoch", "amount0", "provider_fee", "venue", "liquidity"]
    df1 = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame(columns=columns)
    
    # V2 swaps
    v2_block_numbers = np.array([swap.blockNumber for swap in v2_swaps])
    df1["epoch"] = v2_block_numbers // epoch_interval
    df1["amount0"] = np.array([swap.amount0In if swap.amount0In != 0 else swap.amount0Out for swap in v2_swaps])
    df1["provider_fee"] = np.full(len(v2_swaps), 0.003)
    df1["venue"] = np.full(len(v2_swaps), "v2_300")
    df1["liquidity"] = np.full(len(v2_swaps), 313003092010242865)  # Hardcoded since value is relatively consistent

    # V3 swaps
    v3_block_numbers = np.array([swap.blockNumber for swap in v3_swaps])
    df2["epoch"] = v3_block_numbers // epoch_interval
    df2["amount0"] = np.abs(np.array([swap.amount0 for swap in v3_swaps]))
    df2["provider_fee"] = np.full(len(v3_swaps), 0.0005)
    df2["venue"] = np.full(len(v3_swaps), "v3_50")
    df2["liquidity"] = np.array([swap.liquidity for swap in v3_swaps])

    return pd.concat([df1, df2])

v2_swaps : list[SwapEventV2] = load_from_csv(f"swaps_ethereum_v2_300_{from_block}_{to_block}.csv", SwapEventV2)
v3_swaps : list[SwapEvent] = load_from_csv(f"swaps_ethereum_v3_50_{from_block}_{to_block}.csv", SwapEvent)

epoch_interval = 1000
df = combine_swaps(v2_swaps, v3_swaps, epoch_interval, switch_block)
df["returns"] = df["amount0"].astype(float) * df["provider_fee"].astype(float) / df["liquidity"].astype(float)
# Group by epoch and venue, sum the returns column
df = df.groupby(["epoch", "venue"]).agg({
    "returns": "sum",
    "amount0": "sum"
}).reset_index()
df["st"] = [0 if epoch < switch_block // epoch_interval or venue == "v2_300" else 1 for epoch, venue in df[["epoch", "venue"]].values]

# Take log of returns and amount0
df["returns"] = np.log(df["returns"])
df["amount0"] = np.log(df["amount0"])

# Factorize epoch and venue
df["epoch"] = pd.factorize(df["epoch"], sort=True)[0] + 1
df["venue"] = pd.factorize(df["venue"], sort=True)[0] + 1

print("DataFrame info after cleaning:")
print(df.info())

# Apply linear regression 
did_clustered_ols = sm.ols(formula='returns ~ st', data=df).fit(cov_type='cluster', cov_kwds={'groups': np.array(df[['epoch', 'venue']])}, use_t=True)
print(did_clustered_ols.summary())
