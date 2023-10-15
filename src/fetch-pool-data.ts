import { execute } from "../.graphclient";
import fs from "fs";

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

const [token0, token1, startDate, endDate, outFile] = process.argv.slice(2);

(async () => {
  const jsonData = await fetchPoolHistoricalData(
    token0,
    token1,
    Date.parse(startDate) / 1000,
    Date.parse(endDate) / 1000
  );
  fs.writeFileSync(`output/${outFile}.json`, jsonData);
})();
