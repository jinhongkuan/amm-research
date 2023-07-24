### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 4767ae09-7b56-4162-a267-79d8896d54de
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.add("JSON")
	Pkg.add("Plots")
	data_file="../output/USDC_WETH_2022_Q3_2023_Q3.json"
end 

# ╔═╡ f084392c-1a50-4ece-97e6-14fa0545d3ea
begin
	using JSON
using Dates
using Plots
using Printf 
# Function to calculate price from sqrtPriceX96 based on the provided formula
function calculate_price(sqrt_ratio_x96::BigInt)::Float64
    # Convert the BigInt input to BigFloat to perform precise arithmetic
    sqrt_ratio_x96_float = big(sqrt_ratio_x96)

    # Calculate the price using the provided formula with BigFloat
    price_bigfloat = (sqrt_ratio_x96_float / big"2"^96)^2

    # Convert the result back to Float64
    price_float64 = Float64(price_bigfloat)

    return price_float64
end

# Load JSON data from file
function load_json_data(file_path)
    data = JSON.parsefile(file_path)
    return data
end

# Generate timeseries line graph for price
function plot_price_timeseries(data)
	timestamps = []
	prices = []

	for record in data
		push!(timestamps,parse(Int,record["timestamp"]))
		push!(prices,calculate_price(parse(BigInt,record["sqrtPriceX96"])))
	end

	perm_asc_timestamp = sortperm(timestamps)
	(timestamps, prices) = timestamps[perm_asc_timestamp], prices[perm_asc_timestamp]

	p = plot(timestamps, prices, xlabel="Date", ylabel="Price", ylim=(minimum(prices), maximum(prices)), xformatter=(x)-> Dates.format(unix2datetime(x), "mm-dd"), title="Mean Price of Swaps")

	p
end


# Load JSON data from a file (provide your JSON file path)
data = load_json_data(data_file)

# Generate and display the timeseries line graph for price
plot_price_timeseries(data)
end

# ╔═╡ 6665ecae-297f-11ee-1642-130f13a800c2
md"
	## Data Source
	Uniswap V3 data is scraped from the official source on TheGraph https://thegraph.com/hosted-service/subgraph/uniswap/uniswap-v3
"

# ╔═╡ 9904b8f1-8966-42db-9cd6-9dd668355ca3
begin
	# Calculate distribution of trade volume going through each fee tier 
	function fee_split_timeseries(data)
		date_feetier_dict = Dict()
		feetiers = Set([])
		date_format = "yyyy-mm-dd"
		timestamps = []
		
		for record in data
			timestamp = parse(Int, record["timestamp"])
			push!(timestamps, timestamp)
		
			day = Dates.format(unix2datetime(timestamp), date_format)
			feetier = parse(Int, record["pool"]["feeTier"])
			push!(feetiers, feetier)
			key = (date=day, fee_tier=feetier)
			val = get!(date_feetier_dict, key, 0)
			date_feetier_dict[key] += val + parse(Float64, record["amountUSD"]) * feetier / 10000
		end 
	
		
		sorted_feetiers = sort(collect(feetiers))
		
		start_date = unix2datetime(minimum(timestamps))
		end_date = unix2datetime(maximum(timestamps))
		date_range = start_date:Day(1):end_date 
	
		
		weights = Float64[get!(date_feetier_dict, (date=Dates.format(day, date_format), fee_tier=feetier), 0) for day in date_range, feetier in sorted_feetiers]
		weights ./= [(x == 0) ? 1.0 : x for x in sum(weights, dims=2)]
		areaplot(date_range, weights, labels=permutedims(sorted_feetiers), title="Fees Tier Share of Daily Tx Volume")
	end 
	fee_split_timeseries(data)
end

# ╔═╡ Cell order:
# ╟─6665ecae-297f-11ee-1642-130f13a800c2
# ╠═4767ae09-7b56-4162-a267-79d8896d54de
# ╠═f084392c-1a50-4ece-97e6-14fa0545d3ea
# ╠═9904b8f1-8966-42db-9cd6-9dd668355ca3
