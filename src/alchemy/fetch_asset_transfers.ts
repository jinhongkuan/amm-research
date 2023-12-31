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

const FEE_LAYER = "0x2008b6c3D07B061A84F790C035c2f6dC11A0be70";
const FEE_VAULT = "0x37a8f295612602f2774d331e562be9e61B83a327";
const UNISWAP_DUTCH_ORDER_REACTOR =
  "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4";
const UNISWAP_DEFAULT = "0x3B9260A928D30F2A069572805FbB6880FC719A19";

(async () => {
  let results: AssetTransfersResult[] = [];
  let pageKey: string | undefined = undefined;
  const assetTransfersParams: AssetTransfersParams = {
    fromBlock: "0x0",
    toAddress: FEE_LAYER,
    excludeZeroValue: true,
    category: [AssetTransfersCategory.ERC20],
    withMetadata: true,
  };

  while (true) {
    let res = (await alchemy.core.getAssetTransfers({
      ...assetTransfersParams,
      pageKey,
    })) as AssetTransfersResponse;

    pageKey = res.pageKey;
    if (res.transfers.length > 0) {
      const transferAndReceipts = (
        await Promise.all(
          res.transfers.map(async (transfer) => {
            // const response = await fetch(
            //   `https://mainnet.infura.io/v3/${process.env.INFURA_API_KEY}`,
            //   {
            //     method: "POST",
            //     headers: {
            //       "Content-Type": "application/json",
            //     },
            //     body: JSON.stringify({
            //       jsonrpc: "2.0",
            //       method: "eth_getTransactionByHash",
            //       params: [transfer.hash],
            //       id: 1,
            //     }),
            //   }
            // );
            // const transaction = await response.json();

            return {
              ...transfer,
              transaction: await alchemy.core.getTransaction(transfer.hash),
            };
          })
        )
      ).filter((x) => !!x);

      results = [...results, ...transferAndReceipts];
      console.log("Gathered", results.length, "results");
    } else {
      break;
    }

    if (!pageKey) break;
  }

  fs.writeFileSync("results.json", JSON.stringify(results, null, 2));
})();
