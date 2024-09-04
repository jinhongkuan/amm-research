/*
This is an example script that utilizes the Alchemy API to retrieve Uniswap v3 data.
*/

import { BigNumber } from "alchemy-sdk";
import { getSDKProvider } from "./alchemy";
import { getBlockNumberByTimestamp } from "./ethers";
import * as Uniswap from "./uniswap";
import fs from "fs";

const groupOrdersBySenders = (
  dataset: (Uniswap.DutchOrderFilledEvent[] | Uniswap.SwapEvent[])[]
) => {
  const senders: {
    [sender: string]: (Uniswap.DutchOrderFilledEvent | Uniswap.SwapEvent)[];
  } = {};
  dataset.forEach((events) => {
    events.forEach((event) => {
      if (!senders[event.sender]) {
        senders[event.sender] = [];
      }
      senders[event.sender].push(event);
    });
  });
  return senders;
};

(async () => {
  const chainId = 137; // Ethereum Mainnet
  const { alchemy, provider } = await getSDKProvider(chainId);

  const startDate = new Date("2023-11-15T00:00:00.000Z");
  const endDate = new Date("2023-12-01T00:00:00.000Z");
  const selectTokenPairs = [
    ["USDC.e", "WBTC"],
    ["WETH", "WBTC"],
  ];

  const tokenPairs = Uniswap.getTokenPairsWithFrontendFees(chainId).filter(
    (tokenPair) =>
      selectTokenPairs.some(
        (selectTokenPair) =>
          (tokenPair.token0.symbol == selectTokenPair[0] &&
            tokenPair.token1.symbol == selectTokenPair[1]) ||
          (tokenPair.token0.symbol == selectTokenPair[1] &&
            tokenPair.token1.symbol == selectTokenPair[0])
      )
  );

  const oneWeekInSeconds = 7 * 24 * 60 * 60;
  const startTimestamp = startDate.getTime() / 1000;
  const endTimestamp = endDate.getTime() / 1000;
  let currentStartTimestamp = startTimestamp;

  while (currentStartTimestamp < endTimestamp) {
    const currentEndTimestamp = Math.min(
      currentStartTimestamp + oneWeekInSeconds,
      endTimestamp
    );
    const startBlock = await getBlockNumberByTimestamp(
      currentStartTimestamp,
      provider
    );
    const endBlock = await getBlockNumberByTimestamp(
      currentEndTimestamp,
      provider
    );

    for (const tokenPair of tokenPairs) {
      const fillEvents =
        (chainId as any) == 1
          ? await Uniswap.getDutchOrderRouterSwapsForTokenPair(
              tokenPair,
              startBlock,
              endBlock,
              { alchemy, provider },
              Uniswap.makeInjectDutchFillOrdersCategory(chainId)
            )
          : [];

      let frontendSwaps = fillEvents.filter(
        (event) => event.category === Uniswap.Category.FRONTEND_FEES
      );

      let routerSwaps = fillEvents.filter(
        (event) => event.category === Uniswap.Category.ROUTER
      );

      const poolInfos = await Uniswap.getAllPoolAddressesForTokenPair(
        tokenPair,
        provider
      );

      const swapEvents = [];
      for (const poolInfo of poolInfos) {
        console.log(
          `Fetching swaps for pool ${poolInfo.poolAddress}, with fee ${poolInfo.feeAmount}`
        );

        const events = await Uniswap.getSwapsForPool(
          poolInfo,
          startBlock,
          endBlock,
          { alchemy, provider },
          Uniswap.makeInjectSwapEventCategory(chainId)
        );
        for (const event of events) {
          swapEvents.push({
            ...event,
            feeAmount: poolInfo.feeAmount,
          });
        }
      }

      const directSwaps = swapEvents.filter(
        (event) => event.category == Uniswap.Category.DIRECT
      );

      frontendSwaps = frontendSwaps.concat(
        swapEvents.filter(
          (event) => event.category == Uniswap.Category.FRONTEND_FEES
        )
      );

      routerSwaps = routerSwaps.concat(
        swapEvents.filter((event) => event.category == Uniswap.Category.ROUTER)
      );

      console.log(`Number of frontend swaps: ${frontendSwaps.length}`);
      console.log(`Number of direct swaps: ${directSwaps.length}`);
      console.log(`Number of router swaps: ${routerSwaps.length}`);

      const allSwaps = frontendSwaps.concat(directSwaps, routerSwaps);
      const fileName = `queries/uniswap-fees/${chainId}/${tokenPair.token0.symbol}-${tokenPair.token1.symbol}-${currentStartTimestamp}-${currentEndTimestamp}.json`;
      fs.writeFileSync(
        fileName,
        JSON.stringify(
          allSwaps,
          (key, value) =>
            typeof value === "bigint"
              ? value.toString()
              : value["hex"]
              ? BigNumber.from(value["hex"]).toString()
              : value,
          2
        )
      );

      console.log(`Wrote to file: ${fileName}`);
      await new Promise((resolve) => setTimeout(resolve, 10000));
      global.gc?.(); // Explicitly suggests garbage collection
    }

    currentStartTimestamp += oneWeekInSeconds;
  }
})();
