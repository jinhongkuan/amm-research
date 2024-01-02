/*
This is an example script that utilizes the Alchemy API to retrieve Uniswap v3 data.
*/

import { getSDKProvider } from "./alchemy";
import { getBlockNumberByTimestamp } from "./ethers";
import * as Uniswap from "./uniswap";

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

  const startDate = new Date("2023-12-04T00:00:00.000Z");
  const endDate = new Date("2023-12-05T00:00:00.000Z");

  const startBlock = await getBlockNumberByTimestamp(
    startDate.getTime() / 1000,
    provider
  );

  const endBlock = await getBlockNumberByTimestamp(
    endDate.getTime() / 1000,
    provider
  );

  const tokenPair = Uniswap.getTokenPairsWithFrontendFees(chainId)[0];

  console.log(
    `Token Pair: ${tokenPair.token0.symbol} - ${tokenPair.token1.symbol}`
  );

  const fillEvents = await Uniswap.getDutchOrderRouterSwapsForTokenPair(
    tokenPair,
    startBlock,
    endBlock,
    { alchemy, provider },
    Uniswap.makeInjectDutchFillOrdersCategory(chainId)
  );

  const frontendSwaps = fillEvents.filter(
    (event) => event.category === Uniswap.Category.FRONTEND_FEES
  );

  let routerSwaps = fillEvents.filter(
    (event) => event.category === Uniswap.Category.ROUTER
  );

  const poolInfos = await Uniswap.getAllPoolAddressesForTokenPair(tokenPair);

  const swapEvents = Array().concat(
    await Promise.all(
      poolInfos.map(
        async (poolInfo) =>
          await Uniswap.getSwapsForPool(
            poolInfo,
            startBlock,
            endBlock,
            { alchemy, provider },
            Uniswap.makeInjectSwapEventCategory(chainId)
          ).then((events) =>
            events.map((event) => ({ ...event, feeAmount: poolInfo.feeAmount }))
          )
      )
    )
  );

  const directSwaps = swapEvents.filter(
    (event) => event.category === Uniswap.Category.DIRECT
  );

  routerSwaps = routerSwaps.concat(
    swapEvents.filter((event) => event.category === Uniswap.Category.ROUTER)
  );

  console.log(`Number of frontend swaps: ${frontendSwaps.length}`);
  console.log(`Number of direct swaps: ${directSwaps.length}`);
  console.log(`Number of router swaps: ${routerSwaps.length}`);

  const senders = groupOrdersBySenders([
    frontendSwaps.map((event) => ({ ...event, tag: "frontendSwap" })),
    directSwaps.map((event) => ({ ...event, tag: "directSwap" })),
    routerSwaps.map((event) => ({ ...event, tag: "routerSwap" })),
  ]);

  // Sort senders by number of events and display top 10
  const sortedSenders = Object.entries(senders).sort(
    (a, b) => b[1].length - a[1].length
  );

  console.log("Top 10 senders:");
  console.log(sortedSenders.slice(0, 10).map((s) => [s[0], s[1].length]));
})();
