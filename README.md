# AMM Research Supplementary Repo

This repository contains source code for data fetching and data analysis used in the \_\_ paper.

## Installation

1. Ensure that [Node.js](https://nodejs.org/en/download/current) and [Julia](https://julialang.org/downloads/) are installed
2. Install relevant node packages:
   `npm install`
3. Our on-chain data is fetched via The Graph Protocol. Obtain their API key [here](https://thegraph.com/studio/apikeys/) and add it to a `.env` file in the root directory:
   `GRAPHPROTOCOL_API_KEY="XXXX"`

## Usage

1. Run the fetcher script to retrieve relevant data:
   `yarn fetch [token0] [token1] [start date] [end date] [output name]`, e.g.
   `yarn fetch 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 2022-07-01 2023-07-01 USDC_WETH_2022_Q3_2023_Q3`
2. Run Julia script to visualize data ([Pluto](https://plutojl.org/) recommended for interactivity)
3. Replace `data_file` with custom generated output from Step 1
