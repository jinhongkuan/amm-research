import { Alchemy, AlchemySettings, Log, Network } from "alchemy-sdk";
import { EventLog, JsonRpcProvider, ethers } from "ethers";

const MAINNETS: Record<number, Network> = {
  1: Network.ETH_MAINNET,
  137: Network.MATIC_MAINNET,
};

export type AlchemyWithProvider = {
  alchemy: Alchemy;
  provider: JsonRpcProvider;
};
export type LogWithDecodedArgs = Log & {
  args: Record<string, any>;
};
/**
 * Returns an Alchemy SDK provider and a JSON RPC provider for a given chain ID.
 *
 * @param chainId - The chain ID to get the providers for.
 * @returns A promise that resolves to an object containing the Alchemy SDK provider and the JSON RPC provider.
 * @throws Will throw an error if the chain ID is not supported.
 */
export const getSDKProvider = async (
  chainId: number
): Promise<AlchemyWithProvider> => {
  if (!MAINNETS[chainId]) {
    throw new Error(`Chain ID ${chainId} not supported`);
  }

  const config: AlchemySettings = {
    apiKey: process.env.ALCHEMY_API_KEY,
    network: MAINNETS[chainId],
    maxRetries: 5,
    requestTimeout: 5000,
  };

  const alchemy = new Alchemy(config);

  const provider = new JsonRpcProvider(
    (await alchemy.config.getProvider()).connection.url
  );

  return {
    alchemy,
    provider,
  };
};

/**
 * Queries a contract for a specific event within a given block range.
 * The block range is hardcoded to 2000 to avoid hitting the Alchemy API limit.
 *
 * @param {ethers.Contract} contract - The contract to query.
 * @param {number} fromBlock - The starting block number for the query.
 * @param {number} toBlock - The ending block number for the query.
 * @param {ethers.ContractEventName} event - The event to query for.
 * @returns {Promise<Array>} - A promise that resolves to an array of transaction events.
 * @throws Will throw an error if the contract query fails.
 */
export const queryContract = async (
  contract: ethers.Contract,
  { alchemy, provider }: AlchemyWithProvider,
  fromBlock: number,
  toBlock: number,
  topics: any,
  fetchOtherTransactionEvents = false,
  interfaces?: ethers.Interface[]
) => {
  // Hardcode the block range to 2000 to avoid hitting the Alchemy API limit
  const blockRange = 2000;
  const txEvents = [];
  const events = [];
  for (let i = fromBlock; i <= toBlock; i += blockRange) {
    const endBlock = Math.min(i + blockRange - 1, toBlock);
    events.push(
      ...(await provider.getLogs({
        fromBlock: i,
        toBlock: endBlock,
        address: contract.getAddress(),
        topics: topics,
      }))
    );
  }

  console.log("Fetched", events.length, "events");
  let txGroupedEvents;
  if (fetchOtherTransactionEvents) {
    let fails = 0;
    const uniqueTxs = [...new Set(events.map((e) => e.transactionHash))];

    txGroupedEvents = (
      await Promise.all(
        uniqueTxs.map(
          async (txHash) =>
            await alchemy.core
              .getTransactionReceipt(txHash)
              .then((receipt) =>
                receipt?.logs.map((log) => {
                  // Cycle through possible interfaces
                  const contractEvent = interfaces
                    ?.map((i) => i.getEvent(log.topics[0]))
                    .find((e) => e);

                  if (!contractEvent) {
                    return null;
                  }

                  const eventLog = new EventLog(
                    log as any,
                    contract.interface,
                    contractEvent
                  );
                  return eventLog;
                })
              )
              .catch((e) => {
                fails += 1;
                return null;
              })
        )
      )
    ).filter((tx) => tx != null) as EventLog[][];
    console.log(
      "Fetched",
      `${txGroupedEvents.length - fails}/${txGroupedEvents.length}`,
      "transactions"
    );
  } else
    txGroupedEvents = Object.values(
      events.reduce((acc, curr) => {
        const txHash = curr.transactionHash;
        if (!acc[txHash]) {
          acc[txHash] = [];
        }

        const contractEvent = interfaces
          ?.map((i) => i.getEvent(curr.topics[0]))
          .find((e) => e);

        if (!contractEvent) {
          return acc;
        }

        const eventLog = new EventLog(
          curr as any,
          contract.interface,
          contractEvent
        );
        acc[txHash].push(eventLog);
        return acc;
      }, {} as Record<string, ethers.EventLog[]>)
    );

  return txGroupedEvents;
};
