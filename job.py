import modal
import sys
import csv
from uni import alchemy_call_batch_blocks, get_swaps, SwapEvent, SwapEventV2, constants, save_to_csv, get_v2_pool_swaps
from uniswap_analysis import plot_liquidity_rolling_op
from datetime import timedelta
app = modal.App("amm-scraper")
vol = modal.Volume.from_name("amm-scraper-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .copy_local_file("abis/uniswap_router.json", "/root/abis/uniswap_router.json")
    .copy_local_file("abis/uniswap_v2_pool.json", "/root/abis/uniswap_v2_pool.json")
    .copy_local_file("constants.json", "/root/constants.json")
)

@app.function(image=image, timeout=24*60*60, concurrency_limit=1, volumes={"/root/data": vol})
def get_uniswap_v3_swaps(chain, fee, token_pair, from_block, to_block):
    print(f"Getting {chain} {fee} {token_pair} swaps from {from_block} to {to_block}")
    try:
        logs = alchemy_call_batch_blocks(get_swaps, SwapEvent, f"/root/data/swaps_{chain}_v3_{token_pair}_{fee}", int(from_block), int(to_block), chain, constants[chain][f"v3_{token_pair}_{fee}"])
        save_to_csv(logs, f"/root/data/swaps_{chain}_v3_{token_pair}_{fee}_{from_block}_{to_block}.csv", SwapEvent)
        return logs
    except Exception as e:
        print(f"Error in get_uniswap_v3_swaps: {e}")
        raise

@app.function(image=image, timeout=24*60*60, concurrency_limit=1, volumes={"/root/data": vol})
def get_uniswap_v2_swaps(chain, fee, token_pair, from_block, to_block):
    print(f"Getting {chain} {fee} {token_pair} swaps from {from_block} to {to_block}")
    try:
        logs = alchemy_call_batch_blocks(get_v2_pool_swaps, SwapEventV2, f"/root/data/swaps_{chain}_v2_{token_pair}_{fee}", int(from_block), int(to_block), chain, constants[chain][f"v2_{token_pair}_{fee}"])
        save_to_csv(logs, f"/root/data/swaps_{chain}_v2_{token_pair}_{fee}_{from_block}_{to_block}.csv", SwapEventV2)
        return logs
    except Exception as e:
        print(f"Error in get_uniswap_v3_swaps: {e}")
        raise

def submit_job(job_name, data):
    process_job = modal.Function.lookup("amm-scraper", job_name)
    call = process_job.spawn(*data)
    return call.object_id

def get_job_result(call_id):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    result = function_call.get(timeout=30)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python job.py <task> [arguments]")
        sys.exit(1)

    task = sys.argv[1]

    if task == "submit":
        if len(sys.argv) < 4:
            print("Usage: python job.py submit <job_name> <arguments...>")
            sys.exit(1)
        job_name = sys.argv[2]
        job_args = sys.argv[3:]
        call_id = submit_job(job_name, job_args)
        print(f"Job submitted. Call ID: {call_id}")
    elif task == "run":
        if len(sys.argv) < 4:
            print("Usage: python job.py run <job_name> <arguments...>")
            sys.exit(1)
        job_name = sys.argv[2]
        job_args = sys.argv[3:]
        
        # Map job names to their non-Modal implementations
        job_functions = {
            "get_uniswap_v3_swaps": lambda *args: alchemy_call_batch_blocks(
                get_swaps, SwapEvent, f"swaps_{args[0]}_v3_{args[2]}_{args[1]}", 
                int(args[3]), int(args[4]), args[0], constants[args[0]][f"v3_{args[2]}_{args[1]}"]),
            "get_uniswap_v2_swaps": lambda *args: alchemy_call_batch_blocks(
                get_v2_pool_swaps, SwapEventV2, f"swaps_{args[0]}_v2_{args[2]}_{args[1]}", 
                int(args[3]), int(args[4]), args[0], constants[args[0]][f"v2_{args[2]}_{args[1]}"])
        }
        
        if job_name not in job_functions:
            print(f"Unknown job name: {job_name}")
            print(f"Available jobs: {', '.join(job_functions.keys())}")
            sys.exit(1)
            
        try:
            # Run the function locally without Modal
            result = job_functions[job_name](*job_args)
            print(f"Job completed successfully")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error running job: {e}")
            sys.exit(1)

    elif task == "plot":
        if len(sys.argv) < 3:
            print("Usage: python job.py plot <plot_function> [additional_args...]")
            sys.exit(1)
        plot_function = sys.argv[2]
        additional_args = sys.argv[3:]
        
        # Load data from CSV
        from data import load_from_csv, SwapEvent
        
        # Map string to function
        from uniswap_analysis import plot_liquidity_rolling_op, plot_swaps_amount_fee_boxplot
        import re
        # Use a switch statement to handle different plot functions
        match plot_function:
            case "plot_liquidity_rolling_op":
                try:
                    swaps = load_from_csv(additional_args[0], SwapEvent)
                    # Extract chain from file name
                    chain_match = re.search(r'swaps_(\w+)_v3', additional_args[0])
                    if chain_match:
                        chain = chain_match.group(1)
                    else:
                        raise ValueError(f"Could not extract chain from filename: {additional_args[0]}")
                    plot_liquidity_rolling_op(swaps, chain, additional_args[1], timedelta(hours=int(additional_args[2])))
                except Exception as e:
                    print(f"Error in plotting: {e}")
                    print(f"Usage for plot_liquidity_rolling_op: {plot_liquidity_rolling_op.__doc__}")
                    sys.exit(1)
            case "plot_swaps_amount_fee_boxplot":
                try:
                    swaps_dict = {}
                    for arg in additional_args:
                        # Extract fee amount and starting block from filename
                        fee_block_match = re.search(r'v\d+_(\d+)|_(\d+)_(\d+)_\d+', arg)
                        if fee_block_match:
                            fee = int(fee_block_match.group(1) or fee_block_match.group(2))
                            start_block = int(fee_block_match.group(3) or fee_block_match.group(2))
                        else:
                            raise ValueError(f"Could not extract fee and starting block from filename: {arg}")
                        swaps = load_from_csv(arg, SwapEvent)
                        swaps_dict[(fee, start_block)] = swaps
                    plot_swaps_amount_fee_boxplot(swaps_dict)
                except Exception as e:
                    print(f"Error in plotting: {e}")
                    print(f"Usage for plot_swaps_amount_fee_boxplot: {plot_swaps_amount_fee_boxplot.__doc__}")
                    sys.exit(1)
            case _:
                print(f"Unknown plot function: {plot_function}")
                sys.exit(1)

    else:
        print(f"Unknown task: {task}")
        sys.exit(1)