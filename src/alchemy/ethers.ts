import { JsonRpcProvider } from "ethers";

/**
 * This function retrieves the block number for a given timestamp.
 * It uses a binary search algorithm to find the block with the closest timestamp.
 * If the exact timestamp is not found, it returns the start block number.
 *
 * @param {number} timestamp - The timestamp to search for.
 * @param {JsonRpcProvider} provider - The JSON-RPC provider to use for the query.
 * @returns {Promise<number>} - A promise that resolves to the block number.
 */
export const getBlockNumberByTimestamp = async (
  timestamp: number,
  provider: JsonRpcProvider
): Promise<number> => {
  const blockNumber = await provider.getBlockNumber();
  let startBlock = 0;
  let endBlock = blockNumber;

  while (startBlock <= endBlock) {
    const midBlock = Math.floor((startBlock + endBlock) / 2);
    const block = await provider.getBlock(midBlock);

    if (!block) {
      throw new Error("Block not found");
    }

    const blockTimestamp = block.timestamp;

    if (blockTimestamp < timestamp) {
      startBlock = midBlock + 1;
    } else if (blockTimestamp > timestamp) {
      endBlock = midBlock - 1;
    } else {
      return midBlock;
    }
  }

  return startBlock;
};
