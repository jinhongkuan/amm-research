# AMM Research Supplementary Repo

This repository contains source code for data fetching and data analysis used in the \_\_ paper.

## Installation

1. Install [Julia](https://julialang.org/downloads/) 
2. Install the [Jupyter Kernel for Julia](https://github.com/JuliaLang/IJulia.jl) 
3. Connect to the Julia kernel on Jupyter/VSCode Jupyter Extension 

## Usage

1. The main notebook is located under `src/notebook.ipynb` 
2. The notebook is sectioned according to DuneAnalytics queries
3. All queries pre-fetched under `queries/` can be manually obtained according to [Dune Docs](https://dune.com/docs/api/api-reference/get-results/execution-results/)
4. All output graphs are situated under `output/` and mirrors the directory structure of `queries` 

## Queries 

All DuneAnalytics queries is publicly accessible. Here are the ones listed in the notebook
- [DEX Fees and Volumes (By Market)](https://dune.com/queries/3087337)
- [DEX Token Pair Volume](https://dune.com/queries/3106546)
- [DEX Token Pair Fees](https://dune.com/queries/3106719)
- [DEX Venue Volume](https://dune.com/queries/3106538)
