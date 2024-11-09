from data import load_from_csv, save_to_csv, SwapEvent, SwapEventV2, AssetTransfer, SwapEventWithFee
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
from matplotlib.lines import Line2D
from uni import combine_csv_files
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
with open('constants.json', 'r') as f:
    constants = json.load(f)


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

def plot_liquidity_rolling_op(swaps: list[SwapEvent], chain,  operation: str = 'mean', window: timedelta = timedelta(hours=24)):
    df = pd.DataFrame(swaps)
    w3 = Web3(Web3.HTTPProvider(constants[chain]["rpc_url"]))

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
    
    # Set the 'date' column as the index
    df_liquidity.set_index('date', inplace=True)
    
    # Calculate rolling liquidity based on the specified operation and window
    valid_operations = {'mean': 'mean', 'max': 'max', 'min': 'min'}
    if operation not in valid_operations:
        raise ValueError("Invalid operation. Choose 'mean', 'max', or 'min'.")
    
    rolling_method = getattr(df_liquidity['liquidity'].rolling(window=window, center=True, min_periods=1), valid_operations[operation])
    df_liquidity['rolling_liquidity'] = rolling_method()

    # Create x and y for plotting
    x = df_liquidity.index
    y = df_liquidity['rolling_liquidity']

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y, color='blue', linewidth=2, label=f'{operation.capitalize()} Liquidity {window.total_seconds()/3600:.1f}h window)')
    ax.set_ylabel('Liquidity')
    ax.legend()
    ax.set_title(f'{operation.capitalize()} Liquidity Over Time')
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # ax.set_ylim(bottom=0)

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    # Ensure the plot is displayed
    plt.show()


def plot_swaps_amount_fee_boxplot(swaps: dict[tuple[int, int], list[SwapEvent]]):
    # Extract fee and start_block from swaps
    fee_start_block_data = {}
    for (fee, start_block), swap_list in swaps.items():
        if fee not in fee_start_block_data:
            fee_start_block_data[fee] = {}
        fee_start_block_data[fee][start_block] = [np.log(abs(float(swap.amount0))) for swap in swap_list if abs(float(swap.amount0)) > 0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for boxplot
    positions = []
    data = []
    colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(fee_start_block_data)))
    legend_elements = []

    for i, (fee, start_block_data) in enumerate(fee_start_block_data.items()):
        for j, (start_block, values) in enumerate(start_block_data.items()):
            if values:  # Only plot non-empty data
                position = i + j * 0.2
                positions.append(position)
                data.append(values)
                color = colors[j]
                bp = ax.boxplot([values], positions=[position], widths=0.15, patch_artist=True)
                
                # Customize the boxplot
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color=color)
                plt.setp(bp['boxes'], facecolor=color)
                
                # Add median value as text
                median = np.median(values)
                ax.text(position, median, f'{median:.2f}', 
                        horizontalalignment='center', verticalalignment='bottom',
                        fontweight='bold', color='black', fontsize=8)
                
                # Add to legend
                if i == 0:  # Only add to legend once per start_block
                    legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Start Block: {start_block}'))

    ax.set_xlabel('Fee Tier')
    ax.set_ylabel('Swap Amount (log)')
    ax.set_title('Distribution of Swap Amounts by Fee Tier and Start Block')

    # Set x-axis ticks to fee tiers
    ax.set_xticks([i + 0.3 for i in range(len(fee_start_block_data))])
    ax.set_xticklabels(fee_start_block_data.keys())

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # Add legend
    ax.legend(handles=legend_elements, title='Start Blocks', loc='upper right')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
# def analysis_1(): 
#     epoch_interval = 1000
#     v2_swaps = combine_csv_files("swaps_arbitrum_v2_300", SwapEventV2, from_block, to_block)
#     v3_swaps = combine_csv_files("swaps_arbitrum_v3_50", SwapEvent, from_block, to_block)
#     df = combine_swaps(v2_swaps, v3_swaps, epoch_interval, switch_block)
#     df["returns"] = df["amount0"].astype(float) * df["provider_fee"].astype(float) / df["liquidity"].astype(float)
#     # Group by epoch and venue, sum the returns column
#     df = df.groupby(["epoch", "venue"]).agg({
#         "returns": "sum",
#         "amount0": "sum"
#     }).reset_index()
#     df["st"] = [0 if epoch < switch_block // epoch_interval or venue == "v2_300" else 1 for epoch, venue in df[["epoch", "venue"]].values]

#     # Take log of returns and amount0
#     df["returns"] = np.log(df["returns"])
#     df["amount0"] = np.log(df["amount0"])

#     # Factorize epoch and venue
#     df["epoch"] = pd.factorize(df["epoch"], sort=True)[0] + 1
#     df["venue"] = pd.factorize(df["venue"], sort=True)[0] + 1

#     print("DataFrame info after cleaning:")
#     print(df.info())

#     # Apply linear regression 
#     did_clustered_ols = sm.ols(formula='returns ~ st', data=df).fit(cov_type='cluster', cov_kwds={'groups': np.array(df[['epoch', 'venue']])}, use_t=True)
#     print(did_clustered_ols.summary())

# def analysis_2():
#     from_block = get_block_by_datetime(datetime.datetime(2023, 9, 1, 0, 0, 0, tzinfo=pytz.utc), chain)
#     to_block = get_block_by_datetime(datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc), chain)

#     v3_swaps = combine_csv_files("swaps_arbitrum_v3_50", SwapEvent, from_block, to_block)
#     plot_liquidity_rolling_op(v3_swaps)


type SwapGroup = tuple[bool, bool, str] # (fee_tier, platform_fees, token_pair)
def compute_columns(w3: Web3, df: pd.DataFrame):
    first_block_timestamp = get_block_timestamp(w3, int(df.iloc[0].blockNumber))
    last_block_timestamp = get_block_timestamp(w3, int(df.iloc[-1].blockNumber))
    
    # Calculate the total time span in seconds
    total_time_span = last_block_timestamp - first_block_timestamp
    total_blocks = df.blockNumber.max() - df.blockNumber.min()

    # Create a function to estimate timestamp for each block
    def estimate_timestamp(block):
        block_fraction = (block - df.blockNumber.min()) / total_blocks
        return int(first_block_timestamp + block_fraction * total_time_span)
    
    df["time"] = df.apply(lambda row: estimate_timestamp(row.blockNumber), axis=1)
    df["week"] = df["time"].apply(lambda x: datetime.datetime.fromtimestamp(x).isocalendar()[1])
    df.set_index("time", inplace=True)
    df["rolling_liquidity"] = df["liquidity"].astype(float).rolling(window=24*60*60, min_periods=1).mean()
    df["volatility"] = df["sqrtPriceX96"].astype(float).rolling(window=24*60*60, min_periods=1).apply(lambda x: x.max() - x.min())
    df["volume"] = np.log(df["amount0"].abs().rolling(window=24*60*60, min_periods=1).sum())
    print("done")
    # Drop NA rows 
    df.dropna(inplace=True)

    return df

def get_cached_combined_df(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series):
    # Generate hash of input parameters
    import hashlib
    import pickle
    import os
    
    param_str = str(sorted([(str(k), len(v)) for k,v in swaps_per_group.items()]))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    cache_file = f'combined_df_cache_{param_hash}.pkl'

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            combined_df = pickle.load(f)
    else:
        print("Computing data and caching results...")
        combined_df = pd.DataFrame()
        for (fee_tier, platform_fees, token_pair), swaps in swaps_per_group.items():
            df = compute_columns(w3, pd.DataFrame(swaps))
            df["fee"] = fee_tier
            df["platform_fees"] = platform_fees
            df["token_pair"] = token_pair
            df["gas_price"] = df["blockNumber"].map(gas_price)
            combined_df = pd.concat([combined_df, df])
            
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(combined_df, f)
        print(f"Results cached to {cache_file}")
    
    return combined_df

def run_weighted_regression(X: pd.DataFrame, y, clusters):
    unique_clusters = clusters.nunique()
    
    # Initialize model
    model = LinearRegression()
    
    # Calculate weights
    weights = 1 / clusters.map(clusters.value_counts())
    weights = weights / weights.sum() * len(weights)
    
    # Fit model
    model.fit(X, y, sample_weight=weights)
    
    # Bootstrap for standard errors
    n_bootstrap = 1000
    bootstrap_coefs = np.zeros((n_bootstrap, X.shape[1]))
    y_series = pd.Series(y)
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[bootstrap_idx]
        y_boot = y_series.iloc[bootstrap_idx]
        weights_boot = weights.iloc[bootstrap_idx]
        
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot, sample_weight=weights_boot)
        bootstrap_coefs[i] = model_boot.coef_
    
    # Calculate statistics
    std_errors = np.std(bootstrap_coefs, axis=0)
    t_stats = model.coef_ / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(X)-len(X.columns)))
    r2_score = model.score(X, y)
    
    return model, std_errors, t_stats, p_values, r2_score, unique_clusters
def print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights):
    print("\nWeighted Linear Regression Results")
    print(f"(Clustered by {unique_clusters} week-token pair groups)")
    print("-" * 100)
    print(f"R-squared: {r2_score:.4f}")
    print("-" * 100)
    print(f"{'Variable':<30} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'P-value':>10} {'Partial R²':>10} {'Signif':>8}")
    print("-" * 100)
    
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        stars = ''
        if p_values[i] < 0.01:
            stars = '***'
        elif p_values[i] < 0.05:
            stars = '**'
        elif p_values[i] < 0.1:
            stars = '*'
            
        # Calculate partial R² for this variable
        other_vars = [col for col in X.columns if col != name]
        model_others = LinearRegression()
        model_others.fit(X[other_vars], y, sample_weight=weights)
        residuals_others = y - model_others.predict(X[other_vars])
        
        model_single = LinearRegression()
        model_single.fit(X[[name]], residuals_others, sample_weight=weights)
        partial_r2 = model_single.score(X[[name]], residuals_others)
        
        print(f"{name:<30} {coef:>12.4e} {std_errors[i]:>12.4e} {t_stats[i]:>10.2f} {p_values[i]:>10.4e} {partial_r2:>10.4f} {stars:>8}")
    print("-" * 100)

def regress_active_liquidity(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series):
    # Get combined dataframe
    combined_df = get_cached_combined_df(w3, swaps_per_group, gas_price)
    
    # Create feature matrix X
    X = pd.get_dummies(combined_df[['fee', 'platform_fees']], drop_first=True)
    X['fee:platform_fees'] = X['fee'] * X['platform_fees']
    X['volatility'] = combined_df['volatility']
    X['volume'] = combined_df['volume']
    
    # Target variable y
    y = np.log(combined_df['rolling_liquidity'])
    
    # Create cluster groups
    clusters = combined_df.groupby(['week', 'token_pair']).ngroup()
    
    # Run regression
    feature_names = X.columns.tolist()
    model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
        X, y, clusters
    )
    
    # Calculate weights for printing
    weights = 1 / clusters.map(clusters.value_counts())
    weights = weights / weights.sum() * len(weights)
    
    # Print results
    print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights)

def regress_active_liquidity_by_trader(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series, cutoff_block: int):
    # Get combined dataframe
    combined_df = get_cached_combined_df(w3, swaps_per_group, gas_price)

    # Separate by trader type 
    # Get all unique traders and their trade frequencies
    trader_frequencies = combined_df.groupby('sender').size()
    median_frequency = trader_frequencies.median()

    # Find traders who traded both before and after cutoff
    traders_before = set(combined_df[combined_df['blockNumber'] < cutoff_block]['sender'])
    traders_after = set(combined_df[combined_df['blockNumber'] >= cutoff_block]['sender'])
    active_traders = traders_before.intersection(traders_after)

    # Classify traders as retail or institutional based on frequency
    trader_types = {}
    for trader in active_traders:
        frequency = trader_frequencies[trader]
        trader_types[trader] = 'institutional' if frequency > median_frequency else 'retail'

    # Filter combined_df for active traders only and add trader type
    combined_df = combined_df[combined_df['sender'].isin(active_traders)]
    combined_df['trader_type'] = combined_df['sender'].map(trader_types)

    # Split data by trader type and run separate regressions
    for trader_type in ['retail', 'institutional']:
        # Filter data for current trader type
        trader_df = combined_df[combined_df['trader_type'] == trader_type]
        
        if len(trader_df) == 0:
            print(f"\nNo data found for {trader_type} traders")
            continue
            
        # Create feature matrix X
        X = pd.get_dummies(trader_df[['fee', 'platform_fees']], drop_first=True)
        X['fee:platform_fees'] = X['fee'] * X['platform_fees']
        X['volatility'] = trader_df['volatility']
        X['volume'] = trader_df['volume']
        
        # Target variable y
        y = np.log(trader_df['rolling_liquidity'])
        
        # Create cluster groups
        clusters = trader_df.groupby(['week', 'token_pair']).ngroup()
        
        # Run regression
        feature_names = X.columns.tolist()
        model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
            X, y, clusters
        )
        
        # Calculate weights for printing
        weights = 1 / clusters.map(clusters.value_counts())
        weights = weights / weights.sum() * len(weights)
        
        # Print results with trader type header
        print(f"\n=== Results for {trader_type} traders ===")
        print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights)

if __name__ == "__main__":
    w3 = Web3(Web3.HTTPProvider("https://arbitrum.gateway.tenderly.co"))
    # Create gas price series with block numbers as index
    block_range = range(126855341, 155687993)
    gas_price = pd.Series(10000000, index=block_range, name='gasPrice')
    
    usdc_weth_50_no_fees = load_from_csv("swaps_arbitrum_v3_usdc_weth_50_126855341_141283870.csv", SwapEvent)
    usdc_weth_300_no_fees = load_from_csv("swaps_arbitrum_v3_usdc_weth_300_126855341_141283870.csv", SwapEvent)
    usdc_weth_50_with_fees = load_from_csv("swaps_arbitrum_v3_usdc_weth_50_141283870_155687993.csv", SwapEvent)
    usdc_weth_300_with_fees = load_from_csv("swaps_arbitrum_v3_usdc_weth_300_141283870_155687993.csv", SwapEvent)
    wbtc_weth_50_no_fees = load_from_csv("swaps_arbitrum_v3_wbtc_weth_50_126855341_141283870.csv", SwapEvent)
    wbtc_weth_300_no_fees = load_from_csv("swaps_arbitrum_v3_wbtc_weth_300_126855341_141283870.csv", SwapEvent)
    wbtc_weth_50_with_fees = load_from_csv("swaps_arbitrum_v3_wbtc_weth_50_141283871_155687993.csv", SwapEvent)
    wbtc_weth_300_with_fees = load_from_csv("swaps_arbitrum_v3_wbtc_weth_300_141283871_155687993.csv", SwapEvent)
    swaps_per_group = {
        (True, False, "usdc_weth"): usdc_weth_50_no_fees,
        (False, False, "usdc_weth"): usdc_weth_300_no_fees,
        (True, True, "usdc_weth"): usdc_weth_50_with_fees,
        (False, True, "usdc_weth"): usdc_weth_300_with_fees,
        (True, False, "wbtc_weth"): wbtc_weth_50_no_fees,
        (False, False, "wbtc_weth"): wbtc_weth_300_no_fees,
        (True, True, "wbtc_weth"): wbtc_weth_50_with_fees,
        (False, True, "wbtc_weth"): wbtc_weth_300_with_fees
    }
    # regress_active_liquidity(w3, swaps_per_group, gas_price)
    regress_active_liquidity_by_trader(w3, swaps_per_group, gas_price, 141283870)