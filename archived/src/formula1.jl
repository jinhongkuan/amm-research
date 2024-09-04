
using StatsModels, DataFrames, GLM, StatsBase, Printf, Statistics;

begin
    include("./input.jl")
    include("./utils.jl")

    token_pairs = ["USDC.e-WETH", "USDC.e-WBTC", "WETH-WBTC"]
    combined_uniswap_data = DataFrame()
    for token_pair in token_pairs
        uniswap_data = load_and_reorder_transaction_data("./queries/uniswap-fees/137", token_pair)
        uniswap_data_df = DataFrame(uniswap_data)
        uniswap_data_df[!, :token_pair] = token_pair
        combined_uniswap_data = vcat(combined_uniswap_data, uniswap_data_df)
    end

    POLYGON_FRONTENDFEES_BLOCKNUMBER = 48810868

    trades_by_traders = group_by_traders_and_sort(combined_uniswap_data)
    trades_by_traders = filter(pair -> present_across_phases(pair[2], POLYGON_FRONTENDFEES_BLOCKNUMBER), trades_by_traders)
    trade_population_metrics = get_population_metrics(values(trades_by_traders))

    # Assign trader type 
    trader_types = Dict()
    for (trader, trades) in trades_by_traders
        trader_types[trader] = trader_type(trades, trade_population_metrics.median_trade_frequency, trade_population_metrics.median_trade_size)
    end

    # Group traders by type
    retail_traders = filter(pair -> trader_types[pair[1]] == "retail", trades_by_traders)
    sophisticated_traders = filter(pair -> trader_types[pair[1]] == "sophisticated", trades_by_traders)

    f = @formula(volume ~ switch + tier + vix)

    vix = sqrt(var(combined_uniswap_data[!, :price]))
    volume = sum([row[:price] > 1 ? row[:amount0] : row[:amount1] for row in eachrow(combined_uniswap_data)])
end