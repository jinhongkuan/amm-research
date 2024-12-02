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
from uni import combine_csv_files, get_frontend_fees, get_global_fees, alchemy_call_batch_blocks
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
    df["daily_trades"] = df["time"].rolling(window=24*60*60, min_periods=1).apply(lambda x: len(x))

    df.set_index("time", inplace=True)
    df["liquidity"] = np.log(df["liquidity"])
    df["rolling_liquidity"] = df["liquidity"].rolling(window=24*60*60, min_periods=1).mean()
    df["price"] = df["sqrtPriceX96"].astype(float).pow(2).apply(lambda x: x / 2**96)
    df["volatility"] = df["price"].rolling(window=24*60*60, min_periods=1).std()
    df["price"] = np.log(df["price"])
    df = df[df["amount0"] != 0]
    df["amount0"] = np.log(df["amount0"].abs().astype(float))
    df["volume"] = df["amount0"].rolling(window=24*60*60, min_periods=1).sum()

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
    
    # Initialize model with fit_intercept=True to include constant term
    model = LinearRegression(fit_intercept=True)
    
    # Calculate weights
    weights = 1 / clusters.map(clusters.value_counts())
    weights = weights / weights.sum() * len(weights)
    # Fit model
    model.fit(X, y, sample_weight=weights)
    
    # Bootstrap for standard errors
    n_bootstrap = 1000
    bootstrap_coefs = np.zeros((n_bootstrap, X.shape[1] + 1))  # +1 for intercept
    y_series = pd.Series(y)
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[bootstrap_idx]
        y_boot = y_series.iloc[bootstrap_idx]
        weights_boot = weights.iloc[bootstrap_idx]
        
        model_boot = LinearRegression(fit_intercept=True)
        model_boot.fit(X_boot, y_boot, sample_weight=weights_boot)
        bootstrap_coefs[i] = [model_boot.intercept_] + list(model_boot.coef_)
    
    # Calculate statistics including intercept
    std_errors = np.std(bootstrap_coefs, axis=0)
    coefs = [model.intercept_] + list(model.coef_)
    t_stats = coefs / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(X)-len(X.columns)-1))
    r2_score = model.score(X, y)
    
    return model, std_errors, t_stats, p_values, r2_score, unique_clusters
def print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights, show_partial_r2=True):
    print(f"\nWeighted Linear Regression Results (n={len(X)} samples)")
    print("-" * 100)
    print(f"R-squared: {r2_score:.4f}")
    print("-" * 100)
    
    if show_partial_r2:
        print(f"{'Variable':<30} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'P-value':>10} {'Partial R²':>10} {'Signif':>8}")
    else:
        print(f"{'Variable':<30} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'P-value':>10} {'Signif':>8}")
    print("-" * 100)

    # Print constant term first
    stars = ''
    if p_values[0] < 0.01:
        stars = '***'
    elif p_values[0] < 0.05:
        stars = '**'
    elif p_values[0] < 0.1:
        stars = '*'

    if show_partial_r2:
        print(f"{'Constant':<30} {model.intercept_:>12.4e} {std_errors[0]:>12.4e} {t_stats[0]:>10.2f} {p_values[0]:>10.4e} {'N/A':>10} {stars:>8}")
    else:
        print(f"{'Constant':<30} {model.intercept_:>12.4e} {std_errors[0]:>12.4e} {t_stats[0]:>10.2f} {p_values[0]:>10.4e} {stars:>8}")
    
    # Print other coefficients
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        stars = ''
        if p_values[i+1] < 0.01:
            stars = '***'
        elif p_values[i+1] < 0.05:
            stars = '**'
        elif p_values[i+1] < 0.1:
            stars = '*'
            
        # Calculate partial R² for this variable only if needed
        partial_r2 = None
        if show_partial_r2:
            other_vars = [col for col in X.columns if col != name]
            model_others = LinearRegression()
            model_others.fit(X[other_vars], y, sample_weight=weights)
            residuals_others = y - model_others.predict(X[other_vars])
            
            model_single = LinearRegression()
            model_single.fit(X[[name]], residuals_others, sample_weight=weights)
            partial_r2 = model_single.score(X[[name]], residuals_others)
            
            print(f"{name:<30} {coef:>12.4e} {std_errors[i+1]:>12.4e} {t_stats[i+1]:>10.2f} {p_values[i+1]:>10.4e} {partial_r2:>10.4f} {stars:>8}")
        else:
            print(f"{name:<30} {coef:>12.4e} {std_errors[i+1]:>12.4e} {t_stats[i+1]:>10.2f} {p_values[i+1]:>10.4e} {stars:>8}")
    print("-" * 100)

def regress_active_liquidity(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series):
    # Get combined dataframe
    combined_df = get_cached_combined_df(w3, swaps_per_group, gas_price)

    # Split data by fee and run separate regressions
    for fee in [True, False]:
        # Filter data for current fee
        fee_df = combined_df[combined_df['fee'] == fee]
        
        if len(fee_df) == 0:
            print(f"\nNo data found for {fee} bps fee")
            continue
            
        # Create feature matrix X
        X = pd.get_dummies(fee_df[['platform_fees']], drop_first=True)
        X['platform_fees'] = X['platform_fees']
        X['volatility'] = fee_df['volatility']
        X['volume'] = fee_df['volume']
        
        # Target variable y
        y = fee_df['rolling_liquidity']
        
        # Create cluster groups
        clusters = fee_df.groupby(['week', 'token_pair']).ngroup()
        
        # Run regression
        feature_names = X.columns.tolist()
        model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
            X, y, clusters
        )
        
        # Calculate weights for printing
        weights = 1 / clusters.map(clusters.value_counts())
        weights = weights / weights.sum() * len(weights)
        
        # Print results with fee header
        print(f"\n=== Results for {50 if fee else 300} bps fee ===")
        print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights)

def regress_daily_volume(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series):
    # Get combined dataframe
    combined_df = get_cached_combined_df(w3, swaps_per_group, gas_price)

    # Split data by fee and run separate regressions 
    for fee in [True, False]:
        # Filter data for current fee
        fee_df = combined_df[combined_df['fee'] == fee]
        
        if len(fee_df) == 0:
            print(f"\nNo data found for {fee} bps fee")
            continue
            
        # Calculate daily volume by summing amount0 within each day
        fee_df['day'] = fee_df.index.map(lambda x: datetime.datetime.fromtimestamp(x).date())
        daily_volume = fee_df.groupby(['day', 'token_pair'])['amount0'].sum().abs().astype(float)
        fee_df['daily_volume'] = fee_df.apply(lambda x: daily_volume.get((x['day'], x['token_pair']), 0), axis=1)
            
        # Create feature matrix X
        X = pd.get_dummies(fee_df[['platform_fees']], drop_first=True)
        X['platform_fees'] = X['platform_fees']
        X['volatility'] = fee_df['volatility']
        
        # Target variable y - log of daily volume
        y = np.log(fee_df['daily_volume'])
        
        # Create cluster groups
        clusters = fee_df.groupby(['week', 'token_pair']).ngroup()
        
        # Run regression
        feature_names = X.columns.tolist()
        model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
            X, y, clusters
        )
        
        # Calculate weights for printing
        weights = 1 / clusters.map(clusters.value_counts())
        weights = weights / weights.sum() * len(weights)
        
        # Print results with fee header
        print(f"\n=== Results for {50 if fee else 300} bps fee ===")
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
    # Split data by fee and trader type and run separate regressions
    for fee in [True, False]:
        for trader_type in ['retail', 'institutional']:
            # Filter data for current fee and trader type
            trader_df = combined_df[
                (combined_df['trader_type'] == trader_type) & 
                (combined_df['fee'] == fee)
            ]
            
            if len(trader_df) == 0:
                print(f"\nNo data found for {fee} bps fee, {trader_type} traders")
                continue
                
            # Create feature matrix X
            X = pd.get_dummies(trader_df[['platform_fees']], drop_first=True)
            X['volatility'] = stats.zscore(trader_df['volatility'])
            X['volume'] = stats.zscore(trader_df['volume'])
            X['platform_fees:volatility'] = X['platform_fees'] * X['volatility']
            X['platform_fees:volume'] = X['platform_fees'] * X['volume']

            
            # Target variable y
            y = trader_df['rolling_liquidity']
     
            # Create cluster groups
            clusters = trader_df.groupby(['token_pair']).ngroup()
            
            # Run regression
            feature_names = X.columns.tolist()
            model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
                X, y, clusters,
            )
            
            # Calculate weights for printing
            weights = 1 / clusters.map(clusters.value_counts())
            weights = weights / weights.sum() * len(weights)
            
            # Print results with fee and trader type header
            print(f"\n=== Results for {50 if fee else 300} bps fee, {trader_type} traders ===")
            print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights, show_partial_r2=False)

def regress_revenue_by_trader(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series, cutoff_block: int):
    print("Regressing revenue by trader")
    # Get combined dataframe
    combined_df = get_cached_combined_df(w3, swaps_per_group, gas_price)

    # Calculate daily revenue as amount0/liquidity summed over each day
    combined_df['revenue'] = combined_df['amount0'].astype(float) / combined_df['liquidity'].astype(float)
    combined_df = combined_df.dropna(subset=['revenue'])
    # Convert timestamp index to datetime for daily grouping
    combined_df.index = pd.to_datetime(combined_df.index, unit='s')
    # Group by date, sender, fee, platform_fees, token_pair and sum revenue
    daily_df = combined_df.groupby([
        pd.Grouper(freq='D'),
        'sender',
        'fee',
        'platform_fees',
        'token_pair',
        'volatility',
        'week'
    ])['revenue'].sum().reset_index()

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

    # Filter daily_df for active traders only and add trader type
    daily_df = daily_df[daily_df['sender'].isin(active_traders)]
    daily_df['trader_type'] = daily_df['sender'].map(trader_types)

    # Split data by fee and trader type and run separate regressions
    for fee in [True, False]:
        for trader_type in ['retail', 'institutional']:
            # Filter data for current fee and trader type
            trader_df = daily_df[
                (daily_df['trader_type'] == trader_type) & 
                (daily_df['fee'] == fee)
            ]
            
            if len(trader_df) == 0:
                print(f"\nNo data found for {fee} bps fee, {trader_type} traders")
                continue
                
            # Create feature matrix X
            X = pd.get_dummies(trader_df[['platform_fees']], drop_first=True)
            X['platform_fees'] = X['platform_fees']
            X['volatility'] = trader_df['volatility']
            
            # Target variable y (log of daily revenue)
            y = np.log(trader_df['revenue'])
            
            # Create cluster groups using week number from index
            clusters = trader_df.groupby(['week', 'token_pair']).ngroup()
            
            # Run regression
            feature_names = X.columns.tolist()
            model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
                X, y, clusters
            )
            
            # Calculate weights for printing
            weights = 1 / clusters.map(clusters.value_counts())
            weights = weights / weights.sum() * len(weights)
            
            # Print results with fee and trader type header
            print(f"\n=== Results for {50 if fee else 300} bps fee, {trader_type} traders ===")
            print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights)

# token_addresses is sorted by token0 first, then token1
def regress_fees_revenue(swaps: list[SwapEvent], fee_collector_address: str, token_addresses: list[str], pool_address: str, from_block: int, to_block: int):
    frontend_fees = get_frontend_fees(swaps, token_addresses, fee_collector_address, from_block, to_block)
    ff0 = np.zeros(to_block - from_block + 1, dtype=float)
    ff1 = np.zeros(to_block - from_block + 1, dtype=float)
    for fee in frontend_fees:
        if fee.raw_contract_address == token_addresses[0]:
            ff0[fee.blockNum - from_block] += fee.value
        else:
            ff1[fee.blockNum - from_block] += fee.value

    gfees0, gfees1, gfees0_total, gfees1_total = get_global_fees(pool_address, from_block, to_block)
    print(f"Total LP fees: {gfees0_total:.6f}, {gfees1_total:.6f}")
    print(f"Total frontend fees collected: {ff0.sum():.6f}, {ff1.sum():.6f}")

def split_by_trader(df: pd.DataFrame, cutoff_block: int):
    # Get all unique traders and their trade frequencies
    trader_frequencies = df.groupby('sender').size()
    median_frequency = trader_frequencies.median()

    # Find traders who traded both before and after cutoff
    traders_before = set(df[df['blockNumber'] < cutoff_block]['sender'])
    traders_after = set(df[df['blockNumber'] >= cutoff_block]['sender'])
    active_traders = traders_before.intersection(traders_after)

    # Classify traders as retail or institutional based on frequency
    trader_types = {}
    for trader in active_traders:
        frequency = trader_frequencies[trader]
        trader_types[trader] = 'institutional' if frequency > median_frequency else 'retail'

    # Filter daily_df for active traders only and add trader type
    filtered_df = pd.DataFrame(df[df['sender'].isin(active_traders)])
    df = filtered_df
    df['trader_type'] = df['sender'].map(trader_types)
    return df

def regress_volume(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series, cutoff_block: int):
    print("Regressing swap amount")
    # Get combined dataframe
    df = get_cached_combined_df(w3, swaps_per_group, gas_price)
    df = split_by_trader(df, cutoff_block)

    for fee in  [True, False]:
        # Filter data for current fee
        fee_df = df[df['fee'] == fee]
        
        X = pd.get_dummies(fee_df[['platform_fees', 'trader_type']], drop_first=True)
        X['volatility'] = stats.zscore(fee_df['volatility'])
        X['volume'] = stats.zscore(fee_df['volume'])
        X['price'] = stats.zscore(fee_df['price'])
        X['rolling_liquidity'] = stats.zscore(fee_df['rolling_liquidity'])
        X['trader_type:platform_fees'] = X['trader_type_retail'] * X['platform_fees']
        X['trader_type:rolling_liquidity'] = X['trader_type_retail'] * X['rolling_liquidity']
        # Target variable y
        y = fee_df['amount0']
        
        # Create cluster groups
        clusters = fee_df.groupby(['token_pair']).ngroup()
        
        # Run regression
        feature_names = X.columns.tolist()
        model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
            X, y, clusters
        )
        
        # Calculate weights for printing
        weights = 1 / clusters.map(clusters.value_counts())
        weights = weights / weights.sum() * len(weights)
        
        # Print results with fee and trader type header
        print(f"\n=== Results for {50 if fee else 300} bps fee ===")
        print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights, show_partial_r2=False)

def regress_trades_per_day(w3: Web3, swaps_per_group: dict[SwapGroup, list[SwapEvent]], gas_price: pd.Series, cutoff_block: int):
    print("Regressing trades per day")
    # Get combined dataframe
    df = get_cached_combined_df(w3, swaps_per_group, gas_price)
    df = split_by_trader(df, cutoff_block)

    for fee in  [True, False]:
        # Filter data for current fee
        fee_df = df[df['fee'] == fee]
        
        X = pd.get_dummies(fee_df[['platform_fees', 'trader_type']], drop_first=True)
        X['volatility'] = stats.zscore(fee_df['volatility'])
        X['volume'] = stats.zscore(fee_df['volume'])
        X['price'] = stats.zscore(fee_df['price'])
        X['rolling_liquidity'] = stats.zscore(fee_df['rolling_liquidity'])
        X['trader_type:platform_fees'] = X['trader_type_retail'] * X['platform_fees']
        X['trader_type:rolling_liquidity'] = X['trader_type_retail'] * X['rolling_liquidity']
        # Target variable y
        y = fee_df['daily_trades']
        
        # Create cluster groups
        clusters = fee_df.groupby(['token_pair']).ngroup()
        
        # Run regression
        feature_names = X.columns.tolist()
        model, std_errors, t_stats, p_values, r2_score, unique_clusters = run_weighted_regression(
            X, y, clusters
        )
        
        # Calculate weights for printing
        weights = 1 / clusters.map(clusters.value_counts())
        weights = weights / weights.sum() * len(weights)
        
        # Print results with fee and trader type header
        print(f"\n=== Results for {50 if fee else 300} bps fee ===")
        print_regression_results(model, std_errors, t_stats, p_values, r2_score, unique_clusters, feature_names, X, y, weights, show_partial_r2=False)

    

import numpy as np
from scipy.integrate import cumulative_trapezoid

def compute_lvr(prices, volatilities, marginal_liquidity_derivative, times):
    """
    Compute the LVR for a swap.
    
    :param prices: Array of price levels over time (P_t).
    :param volatilities: Array of volatility levels
    :param marginal_liquidity_derivative: Array of marginal liquidity derivatives 
    :param times: Array of time points corresponding to the prices and volatilities.
    :return: Array of cumulative LVR values over time.
    """
    # Ensure arrays are numpy arrays for vectorized operations
    prices = np.array(prices)
    volatilities = np.array(volatilities)
    marginal_liquidity_derivative = np.array(marginal_liquidity_derivative)
    times = np.array(times)

    # Compute instantaneous LVR
    instantaneous_lvr = 0.5 * (volatilities ** 2) * (prices ** 2) * marginal_liquidity_derivative

    # Compute cumulative LVR using numerical integration
    cumulative_lvr = cumulative_trapezoid(instantaneous_lvr, times, initial=0)
    
    return cumulative_lvr


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
    # # regress_active_liquidity(w3, swaps_per_group, gas_price)
    # # regress_daily_volume(w3, swaps_per_group, gas_price)
    # regress_active_liquidity_by_trader(w3, swaps_per_group, gas_price, 141283870)
    # regress_fees_revenue(arbitrum_weth_usdc_50, constants["arbitrum"]["fee_collector"], [constants["arbitrum"]["tokens"]["USDC"], constants["arbitrum"]["tokens"]["WETH"]], constants["arbitrum"]["v3_usdc_weth_50"], 143855317, 155687993)
    regress_volume(w3, swaps_per_group, gas_price, 141283870)
    # regress_trades_per_day(w3, swaps_per_group, gas_price, 141283870)