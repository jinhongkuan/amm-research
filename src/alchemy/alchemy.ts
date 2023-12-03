import { Alchemy, Log, Network } from "alchemy-sdk";
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

  const config = {
    apiKey: process.env.ALCHEMY_API_KEY,
    network: MAINNETS[chainId],
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
  alchemy: Alchemy,
  fromBlock: number,
  toBlock: number,
  event: ethers.ContractEventName,
  fetchOtherTransactionEvents = false
) => {
  // Hardcode the block range to 2000 to avoid hitting the Alchemy API limit
  const blockRange = 2000;
  const txEvents = [];
  for (let i = fromBlock; i <= toBlock; i += blockRange) {
    const endBlock = Math.min(i + blockRange - 1, toBlock);
    const events = await contract.queryFilter(event, i, endBlock);
    const txGroupedEvents = fetchOtherTransactionEvents
      ? ((
          await Promise.all(
            [...new Set(events.map((e) => e.transactionHash))].map(
              async (txHash) =>
                await alchemy.core
                  .getTransactionReceipt(txHash)
                  .then((receipt) =>
                    receipt?.logs.map(
                      (log) =>
                        new EventLog(
                          log as any,
                          contract.interface,
                          contract.interface.getEvent(log.topics[0])!
                        )
                    )
                  )
            )
          )
        ).filter((tx) => !!tx) as EventLog[][])
      : Object.values(
          events.reduce((acc, curr) => {
            const txHash = curr.transactionHash;
            if (!acc[txHash]) {
              acc[txHash] = [];
            }
            if (curr instanceof EventLog) {
              if (!curr.args) {
                console.log(curr);
              }
              acc[txHash].push(curr);
            }
            return acc;
          }, {} as Record<string, ethers.EventLog[]>)
        );

    txEvents.push(...txGroupedEvents);
  }
  return txEvents;
};
