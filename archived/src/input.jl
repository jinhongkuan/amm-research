using JSON

function load_and_reorder_transaction_data(folder_name::String, file_substring::String)
    uniswap_fees_data = []
    for file in readdir(folder_name, join=true)
        if occursin(file_substring, file)
            append!(uniswap_fees_data, JSON.parsefile(file))
        end
    end
    reorder_amount(uniswap_fees_data)
    return uniswap_fees_data
end

# This function is used to reorder the amount0 and amount1 fields based on heuristics
function reorder_amount(array::Array)
    array = filter(entry -> entry["price"] != 0, array)
    array = filter(entry -> haskey(entry, "blockNumber"), array)

    for entry in array
        if abs(parse(BigInt, entry["amount0"])) > abs(parse(BigInt, entry["amount1"]))
            entry["amount0"], entry["amount1"] = entry["amount1"], entry["amount0"]
            entry["price"] = 1 / entry["price"]
        end
    end
end
