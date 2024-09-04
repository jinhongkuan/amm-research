function trader_type(sorted_history, freq_threshold, size_threshold)
    # Evaluate trade frequency and amount 
    trade_frequency = length(sorted_history)
    trade_amount = sum(abs(parse(BigInt, t["amount0"])) for t in sorted_history)
    trade_size = trade_amount / trade_frequency

    if trade_frequency > freq_threshold && trade_size > size_threshold
        return "sophisticated"
    else
        return "retail"
    end
end

function get_population_metrics(histories)
    # Get trade frequency and amount for each trader
    trade_frequency = [length(history) for history in histories]
    trade_amount = [sum(abs(parse(BigInt, t["amount0"])) for t in history) for history in histories]
    trade_size = trade_amount ./ trade_frequency

    # Get mean and standard deviation of trade frequency and amount

    mean_trade_frequency = mean(trade_frequency)
    std_trade_frequency = std(trade_frequency)
    median_trade_frequency = median(trade_frequency)

    mean_trade_size = mean(trade_size)
    std_trade_size = std(trade_size)
    median_trade_size = median(trade_size)

    return (
        mean_trade_frequency=mean_trade_frequency,
        std_trade_frequency=std_trade_frequency,
        median_trade_frequency=median_trade_frequency,
        mean_trade_size=mean_trade_size,
        std_trade_size=std_trade_size,
        median_trade_size=median_trade_size
    )
end

function group_by_traders_and_sort(trades)
    # Group by sender
    trades_by_trader = Dict()
    for trade in trades
        sender = trade["sender"]
        if haskey(trades_by_trader, sender)
            push!(trades_by_trader[sender], trade)
        else
            trades_by_trader[sender] = [trade]
        end
    end

    # Sort trades by blockNumber
    for (trader, trades) in trades_by_trader
        trades_by_trader[trader] = sort(trades, by=entry -> entry["blockNumber"])
    end

    return trades_by_trader
end

function split_by_phase(sorted_history, phase_switch)
    # Split into two phases
    try
        phase_1 = filter(row -> row["blockNumber"] < phase_switch, sorted_history)
        phase_2 = filter(row -> row["blockNumber"] >= phase_switch, sorted_history)

        return phase_1, phase_2

    catch
    end

    return [], []
end