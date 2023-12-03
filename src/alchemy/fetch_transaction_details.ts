// Setup: npm install alchemy-sdk
import {
  Alchemy,
  AssetTransfersCategory,
  AssetTransfersParams,
  AssetTransfersResponse,
  AssetTransfersResult,
  Network,
  TransactionReceipt,
} from "alchemy-sdk";
import fs from "fs";

const config = {
  apiKey: process.env.ALCHEMY_API_KEY,
  network: Network.ETH_MAINNET,
};
const alchemy = new Alchemy(config);

// Read from input json file
const filePath = process.argv[2];
if (!filePath) {
  console.error("Please provide a file path as the first argument.");
  process.exit(1);
}
let fileContents;
try {
  fileContents = fs.readFileSync(filePath, "utf8");
} catch (err) {
  console.error("Error reading file:", err);
  process.exit(1);
}
const txHashes = JSON.parse(fileContents);

(async () => {
  const receipts = (
    await Promise.all<TransactionReceipt | null>(
      txHashes.map(
        async (txHash: string) =>
          await alchemy.core.getTransactionReceipt(txHash)
      )
    )
  ).filter((x) => !!x) as TransactionReceipt[];
  console.log(receipts[0].logs);
})();
