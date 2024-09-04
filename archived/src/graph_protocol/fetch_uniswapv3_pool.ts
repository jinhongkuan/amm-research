import { execute } from "../../.graphclient";
import fs from "fs";
import { program } from "commander"; 

export const fetchPoolHistoricalData = async (
  token0: string,
  token1: string,
  startDate: number,
  endDate: number
) => {
  let page = 0;
  let pageSize = 100; // Adjust the page size as needed
  let allSwaps = [];

  while (true) {
    const query = `
      query {
        swaps(
          first: ${pageSize}
          skip: ${page * pageSize}
          where: {
            token0: "${token0}"
            token1: "${token1}"
            timestamp_gt: ${startDate}
            timestamp_lt: ${endDate}
          }
        ) {
          transaction {
            gasUsed
            gasPrice
          }
          pool {
            feeTier
          }
          amount0
          amount1
          amountUSD
          sqrtPriceX96
          timestamp
        }
      }
    `;

    const result = await execute(
      query,
      {},
      {
        config: {
          apiToken: process.env.GRAPHPROTOCOL_API_KEY,
        },
      }
    );

    if (!result.data?.swaps || result.data.swaps.length === 0) {
      break; // No more data to fetch, exit the loop
    }

    allSwaps.push(...result.data.swaps);
    page++;
  }

  // Save the combined data as a JSON object
  const jsonData = JSON.stringify(allSwaps, null, 2);

  return jsonData;
};

program
  .description('Fetch historical data for a Uniswap V3 pool')
  .argument('<token0>', 'The address of the first token in the pool')
  .argument('<token1>', 'The address of the second token in the pool')
  .argument('<start-date>', 'The start date for the data fetch in YYYY-MM-DD format')
  .argument('<end-date>', 'The end date for the data fetch in YYYY-MM-DD format')
  .option('-o, --out-file <type>', 'Name of the output file where the fetched data will be stored')
  .action((token0: string, token1: string, startDate: string, endDate: string, options: any) => {
    const outFile =  `queries/graph_protocol/uniswapv3_pool_fees/${options.outFile ?? "uniswapv3_pool"}.json`;

    if (!token0 || !token1 || !startDate || !endDate) {
      program.help();
    }

    (async () => {
      const jsonData = await fetchPoolHistoricalData(
        token0.toLowerCase(),
        token1.toLowerCase(),
        Date.parse(startDate) / 1000,
        Date.parse(endDate) / 1000
      );
      fs.writeFileSync(outFile, jsonData);
    })();

    
  })
  .showHelpAfterError()
  .parse(process.argv);

