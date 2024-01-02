import { Token, ChainId, Price, CurrencyAmount } from "@uniswap/sdk-core";
import { Pool, FeeAmount } from "@uniswap/v3-sdk";
import TOKENS_WITH_FEES from "./constants/tokens_with_fees.json";
import USD_PEGGED_TOKENS from "./constants/usd_pegged_tokens.json";
import { BigNumber } from "alchemy-sdk";
import IUniswapV3PoolArtifact from "@uniswap/v3-core/artifacts/contracts/interfaces/IUniswapV3Pool.sol/IUniswapV3Pool.json";
import UniswapXDutchOrderRouterABI from "./interfaces/UniswapXDutchOrderRouterABI.json";
import ERC20_ABI from "./interfaces/ERC20ABI.json";
import FiatTokenABI from "./interfaces/FiatTokenABI.json";
import WETH_ABI from "./interfaces/WETHABI.json";

import { EventLog, ethers } from "ethers";
import addresses from "./constants/addresses.json";
import * as Alchemy from "./alchemy";

export type TokenPair = {
  token0: Token;
  token1: Token;
};

export type PoolInfo = TokenPair & {
  poolAddress: string;
  feeAmount: FeeAmount;
};

export type SwapEvent = {
  sender: string;
  recipient: string;
  amount0: BigNumber;
  amount1: BigNumber;
  sqrtPriceX96: BigNumber;
  liquidity: string;
  tick: number;
  hash: string;
};

export type DutchOrderFilledEvent = {
  sender: string;
  recipient: string;
  amount0: BigNumber;
  amount1: BigNumber;
  hash: string;
};

export enum Category {
  DIRECT,
  ROUTER,
  FRONTEND_FEES,
}

export type eventsProcessor<T> = (events: EventLog[]) => T;

/**
 * This function returns all possible token pairs from the TOKENS_WITH_FEES list that are on the specified chainId.
 * If both tokens in a pair are USD pegged, the pair is skipped.
 * @param {ChainId} chainId - The chainId to filter the tokens by.
 * @returns {TokenPair[]} - An array of token pairs.
 */
export const getTokenPairsWithFrontendFees = (chainId: ChainId) => {
  const tokensFromChain = TOKENS_WITH_FEES.filter(
    (token) => token.chainId === chainId
  );

  const tokenObjectToToken = (token: any) => {
    return new Token(
      token.chainId,
      (token.address as string).toLowerCase(),
      token.decimals,
      token.symbol,
      token.name
    );
  };

  // Get all pairings of different tokens
  const tokenPairs = tokensFromChain
    .flatMap((token, index) => {
      return tokensFromChain.slice(index + 1).map((token1) => {
        // If both tokens are USD pegged, skip
        if (
          USD_PEGGED_TOKENS.includes(token.address) &&
          USD_PEGGED_TOKENS.includes(token1.address)
        ) {
          return null;
        }

        return {
          token0: tokenObjectToToken(token),
          token1: tokenObjectToToken(token1),
        };
      });
    })
    .filter((x) => !!x) as TokenPair[];

  return tokenPairs;
};

/**
 * This function returns all pool addresses for a given token pair.
 * It maps over all possible fee amounts and generates a pool address for each.
 *
 * @param {TokenPair} tokenPair - The token pair to get pool addresses for.
 * @returns {PoolInfo[]} - An array of pool information, including the pool address and fee amount.
 */
export const getAllPoolAddressesForTokenPair = (
  tokenPair: TokenPair
): PoolInfo[] => {
  return Array(
    FeeAmount.LOWEST,
    FeeAmount.LOW,
    FeeAmount.MEDIUM,
    FeeAmount.HIGH
  ).map((feeAmount) => {
    const address = Pool.getAddress(
      tokenPair.token0,
      tokenPair.token1,
      feeAmount
    );

    return {
      token0: tokenPair.token0,
      token1: tokenPair.token1,
      feeAmount,
      poolAddress: address,
    };
  });
};

/**
 * This function retrieves swap events for a given pool within a specified block range.
 * It creates a new contract instance with the pool's address and ABI, then queries for Swap events.
 * Each event is mapped to a SwapEvent object, and null events are filtered out.
 *
 * @param {PoolInfo} pool - The pool to get swap events for.
 * @param {string} fromBlock - The starting block number for the query.
 * @param {string} toBlock - The ending block number for the query.
 * @param {JsonRpcProvider} provider - The JSON-RPC provider to use for the query.
 * @returns {Promise<SwapEvent[]>} - A promise that resolves to an array of SwapEvent objects.
 */
export const getSwapsForPool = async <T = undefined>(
  pool: PoolInfo,
  fromBlock: number,
  toBlock: number,
  { alchemy, provider }: Alchemy.AlchemyWithProvider,
  eventProcessor?: eventsProcessor<T>
): Promise<(SwapEvent & T)[]> => {
  const poolContract = new ethers.Contract(
    pool.poolAddress,
    IUniswapV3PoolArtifact.abi,
    provider
  );

  const txEvents = await Alchemy.queryContract(
    poolContract,
    { alchemy, provider },
    fromBlock,
    toBlock,
    ["0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"],
    false,
    [new ethers.Interface(IUniswapV3PoolArtifact.abi)]
  );

  return txEvents
    .map((events) => {
      const swapEvent = events.find((event) => event.eventName === "Swap")!;

      const { args } = swapEvent;

      const swapEventContent = {
        hash: swapEvent.transactionHash,
        sender: args.sender,
        recipient: args.recipient,
        amount0: args.amount0,
        amount1: args.amount1,
        sqrtPriceX96: args.sqrtPriceX96,
        price: convertSqrtPriceX96ToPrice(
          args.sqrtPriceX96,
          pool.token0,
          pool.token1
        ),
        liquidity: args.liquidity,
        tick: args.tick,
      } as SwapEvent;

      return eventProcessor
        ? { ...swapEventContent, ...eventProcessor(events) }
        : swapEventContent;
    })
    .filter((x) => !!x) as (SwapEvent & T)[];
};

/**
 * Fetches Dutch Order Router Swaps for a given token pair within a specified block range.
 *
 * @param {TokenPair} tokenPair - The token pair to fetch swaps for.
 * @param {number} fromBlock - The starting block number for the query.
 * @param {number} toBlock - The ending block number for the query.
 * @param {JsonRpcProvider} provider - The JSON-RPC provider to use for the query.
 * @returns {Promise<DutchOrderFilledEvent[]>} - A promise that resolves to an array of Dutch Order Filled Events.
 */
export const getDutchOrderRouterSwapsForTokenPair = async <T = undefined>(
  tokenPair: TokenPair,
  fromBlock: number,
  toBlock: number,
  { alchemy, provider }: Alchemy.AlchemyWithProvider,
  eventProcessor?: eventsProcessor<T>
): Promise<(DutchOrderFilledEvent & T)[]> => {
  const chainId = tokenPair.token0.chainId as unknown as keyof typeof addresses;

  const token0Contract = new ethers.Contract(
    tokenPair.token0.address,
    ERC20_ABI,
    provider
  );

  const contractInterfaces = [
    ERC20_ABI,
    IUniswapV3PoolArtifact.abi,
    UniswapXDutchOrderRouterABI,
    FiatTokenABI,
    WETH_ABI,
  ].map((abi) => new ethers.Interface(abi as any));

  const txEvents0 = await Alchemy.queryContract(
    token0Contract,
    { alchemy, provider },
    fromBlock,
    toBlock,
    [
      ethers.id("Transfer(address,address,uint256)"),
      [ethers.zeroPadValue(addresses[chainId].FEE_LAYER, 32)],
      null,
    ],
    true,
    contractInterfaces
  );

  const txEvents1 = await Alchemy.queryContract(
    token0Contract,
    { alchemy, provider },
    fromBlock,
    toBlock,
    [
      ethers.id("Transfer(address,address,uint256)"),
      null,
      [ethers.zeroPadValue(addresses[chainId].FEE_LAYER, 32)],
    ],
    true,
    contractInterfaces
  );

  const txEvents = [...txEvents0, ...txEvents1];

  return txEvents
    .map((events) => {
      const transferTo = events.find(
        (event) =>
          event instanceof EventLog &&
          event.eventName === "Transfer" &&
          event.args[1] == addresses[chainId].FEE_LAYER
      )! as any;

      const transferFrom = events.find(
        (event) =>
          event instanceof EventLog &&
          event.eventName === "Transfer" &&
          event.args[0] == addresses[chainId].FEE_LAYER
      )! as any;
      if (!transferTo || !transferFrom) return null;

      if (
        !(
          (transferTo.address == tokenPair.token0.address &&
            transferFrom.address == tokenPair.token1.address) ||
          (transferTo.address == tokenPair.token1.address &&
            transferFrom.address == tokenPair.token0.address)
        )
      ) {
        return null;
      }

      const flipTokenOrder = tokenPair.token0.sortsBefore(tokenPair.token1);

      const fillEventContent = {
        hash: transferFrom.transactionHash,
        sender: transferTo.args[0],
        recipient: transferFrom.args[1],
        amount0: flipTokenOrder
          ? BigNumber.from(transferFrom.data)
          : -BigNumber.from(transferTo.data),
        amount1: flipTokenOrder
          ? -BigNumber.from(transferTo.data)
          : BigNumber.from(transferFrom.data),
        price: 1,
      } as DutchOrderFilledEvent;

      return eventProcessor
        ? { ...eventProcessor(events), ...fillEventContent }
        : fillEventContent;
    })
    .filter((x) => !!x) as (DutchOrderFilledEvent & T)[];
};

/**
 * This function injects a SwapCategory into a SwapEvent based on the event address.
 * If the event address matches the FEE_LAYER address, the category is set to FEE_LAYER.
 * If the event address matches the UNISWAP_UNIVERSAL_ROUTER address, the category is set to UNIVERSAL_ROUTER.
 * If the event address does not match any of the above, the category is set to DIRECT.
 *
 * @param {Log | EventLog[]} events - The events to search for a matching address.
 * @param {SwapEvent} swapEvent - The SwapEvent to inject the category into.
 * @param {number} chainId - The chainId to use for looking up the addresses.
 * @returns {SwapEvent & { category: Category }} - The SwapEvent with the injected category.
 */
export const makeInjectSwapEventCategory =
  (chainId: number) =>
  (events: EventLog[]): { category: Category } => {
    const _chainId = chainId as unknown as keyof typeof addresses;

    if (
      events.find(
        (event) =>
          event instanceof EventLog &&
          event.eventName === "Transfer" &&
          event.args[0] == addresses[_chainId].FEE_VAULT
      )
    ) {
      return {
        category: Category.FRONTEND_FEES,
      };
    }

    if (
      events.find(
        (event) =>
          event instanceof EventLog &&
          event.eventName === "Swap" &&
          (event.args[0] === addresses[_chainId].UNISWAP_UNIVERSAL_ROUTER ||
            event.args[0] === (addresses[_chainId] as any).UNISWAP_ROUTER_V2)
      )
    ) {
      return {
        category: Category.ROUTER,
      };
    }

    return {
      category: Category.DIRECT,
    };
  };

export const makeInjectDutchFillOrdersCategory =
  (chainId: number) =>
  (events: EventLog[]): { category: Category } => {
    const _chainId = chainId as unknown as keyof typeof addresses;

    if (
      events.find(
        (event) =>
          event instanceof EventLog &&
          event.eventName === "Transfer" &&
          event.args[0] == addresses[_chainId].FEE_VAULT
      )
    ) {
      return {
        category: Category.FRONTEND_FEES,
      };
    }

    return {
      category: Category.ROUTER,
    };
  };

/**
 * This function converts sqrtPriceX96 into a floating point price.
 * It accounts for the differing decimals between the two tokens.
 *
 * @param {string} sqrtPriceX96 - The sqrtPriceX96 to convert.
 * @param {Token} token0 - The first token in the pair.
 * @param {Token} token1 - The second token in the pair.
 * @returns {number} - The converted price.
 */
export const convertSqrtPriceX96ToPrice = (
  sqrtPriceX96: string,
  token0: Token,
  token1: Token
): number => {
  const PRICE_SCALE = BigNumber.from(2).pow(96);
  const sqrtPrice = BigNumber.from(sqrtPriceX96).div(PRICE_SCALE);
  const priceQ192 = sqrtPrice.mul(sqrtPrice);
  const tokenBase = token0.sortsBefore(token1) ? token0 : token1;
  const tokenQuote = token0.sortsBefore(token1) ? token1 : token0;
  const price = new Price({
    baseAmount: CurrencyAmount.fromRawAmount(tokenBase, 1),
    quoteAmount: CurrencyAmount.fromRawAmount(tokenQuote, priceQ192.toString()),
  });
  return parseFloat(price.toSignificant(6));
};
