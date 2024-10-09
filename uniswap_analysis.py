from data import load_from_csv, save_to_csv, SwapEvent, SwapEventV2, AssetTransfer
from utils import get_block_by_datetime
import datetime
import json
import pandas as pd
import pytz
import statsmodels.formula.api as sm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_block_timestamp
from web3 import Web3
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

chain = "arbitrum"
url = f"https://arbitrum.gateway.tenderly.co"
w3 = Web3(Web3.HTTPProvider(url))
# from_block = get_block_by_datetime(datetime.datetime(2023, 9, 1, 0, 0, 0, tzinfo=pytz.utc), chain)
# switch_block = get_block_by_datetime(datetime.datetime(2023, 10, 17, 0, 0, 0, tzinfo=pytz.utc), chain)
# to_block = get_block_by_datetime(datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc), chain)
from_block = 18037988
to_block = 18687851

def match_swap_asset_transfer(swap: SwapEvent, asset_transfer: AssetTransfer) -> bool:
    return swap.transactionHash.lower() == asset_transfer.hash.lower()

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

def plot_liquidity(swaps: list[SwapEvent]):
    df = pd.DataFrame(swaps)

    # Average out liquidity within the same block 
    # Group by block and average out liquidity
    df_grouped = df.groupby('blockNumber')['liquidity'].mean().reset_index()
    
    # Sort by block number to ensure chronological order
    df_grouped = df_grouped.sort_values('blockNumber')
    # Create x and y for plotting, dropping any NA values
    df_grouped_clean = df_grouped.dropna(subset=['blockNumber', 'liquidity'])
    df_grouped_clean = df_grouped_clean[df_grouped_clean['liquidity'] != 0]
    x = df_grouped_clean['blockNumber']
    y = df_grouped_clean['liquidity']
    # Get the timestamps for the first and last block
    first_block_timestamp = get_block_timestamp(w3, swaps[0].blockNumber)
    last_block_timestamp = get_block_timestamp(w3, swaps[-1].blockNumber)
    
    # Calculate the total time span in seconds
    total_time_span = last_block_timestamp - first_block_timestamp
    total_blocks = x.max() - x.min()
    
    # Create a function to estimate timestamp for each block
    def estimate_timestamp(block):
        block_fraction = (block - x.min()) / total_blocks
        return first_block_timestamp + block_fraction * total_time_span
    
    # Create a new DataFrame with estimated timestamps
    df_liquidity = pd.DataFrame({
        'blockNumber': x,
        'liquidity': y.astype(float),
        'estimated_timestamp': x.apply(estimate_timestamp)
    })
    # Convert estimated timestamps to datetime objects
    df_liquidity['date'] = pd.to_datetime(df_liquidity['estimated_timestamp'], unit='s')
    
    # Calculate rolling average liquidity using a 24-hour window
    df_liquidity['rolling_liquidity'] = df_liquidity['liquidity'].rolling(window=3600*24, center=True, min_periods=1).mean()
    df_liquidity['max_hourly_liquidity'] = df_liquidity['liquidity'].rolling(window=60*5, center=True, min_periods=1).max()
    # Create x and y for plotting
    x = df_liquidity['date']
    y = df_liquidity['liquidity']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # First subplot: Max Hourly Liquidity
    ax1.fill_between(x, df_liquidity["max_hourly_liquidity"], color='blue', alpha=0.3, label='Max Liquidity (5m)')
    ax1.set_ylabel('Max Liquidity')
    ax1.legend()
    ax1.set_title('Max Liquidity Over Time')
    ax1.set_xlim(x.min(), x.max())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.set_ylim(bottom=0)

    # Second subplot: Rolling Average
    ax2.plot(x, df_liquidity["rolling_liquidity"], color='red', linewidth=2, label='Rolling Average (24h)', linestyle='-')
    ax2.set_ylabel('Rolling Average Liquidity')
    ax2.legend()
    ax2.set_title('24-hour Rolling Average Liquidity Over Time')
    ax2.set_xlim(x.min(), x.max())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.set_ylim(bottom=0)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    # Ensure the plot is displayed
    plt.show()

    

v2_swaps : list[SwapEventV2] = load_from_csv(f"swaps_ethereum_v2_300_{from_block}_{to_block}.csv", SwapEventV2)
v3_swaps : list[SwapEvent] = load_from_csv(f"swaps_ethereum_v3_50_{from_block}_{to_block}.csv", SwapEvent)

def analysis_1(): 
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

def analysis_2():
    plot_liquidity(v3_swaps)

analysis_2()