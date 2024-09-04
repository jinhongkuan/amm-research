# AMM Research Supplementary Repo

This repository contains source code for data fetching and data analysis used in the \_\_ paper.

## Installation

1. Install [Julia](https://julialang.org/downloads/) 
2. Install the [Jupyter Kernel for Julia](https://github.com/JuliaLang/IJulia.jl) 
3. Connect to the Julia kernel on Jupyter/VSCode Jupyter Extension 

## Usage


1. The main notebook is located under `src/notebook.ipynb` 
2. The notebook is sectioned according to DuneAnalytics and TheGraph queries
3. All queries pre-fetched under `queries/` can be manually obtained via [Dune](https://dune.com/docs/api/api-reference/get-results/execution-results/) or [The Graph]()
4. All output graphs are situated under `output/` and mirrors the directory structure of `queries` 

## Queries 

### Dune Analytics
All Dune Analytics queries is publicly accessible. Here are the ones listed in the notebook
- [DEX Fees and Volumes (By Market)](https://dune.com/queries/3087337)
- [DEX Token Pair Volume](https://dune.com/queries/3106546)
- [DEX Token Pair Fees](https://dune.com/queries/3106719)
- [DEX Venue Volume](https://dune.com/queries/3106538)

### The Graph 
3. Our on-chain data is fetched via The Graph Protocol. Obtain their API key [here](https://thegraph.com/studio/apikeys/) and add it to a `.env` file in the root directory:
   `GRAPHPROTOCOL_API_KEY="XXXX"`

1. Run the fetcher script to retrieve relevant data:
   `yarn fetch [token0] [token1] [start date] [end date] [output name]`, e.g.
   `ts-node src/thegraph/fetch_uniswapv3_pool.ts 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2 2022-10-01 2023-10-01`