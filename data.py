from dataclasses import dataclass
from typing import Dict, Any
from decimal import Decimal
from dataclass_csv import DataclassWriter, DataclassReader

@dataclass
class AssetTransfer:
    blockNum: int
    uniqueId: str
    hash: str
    from_address: str
    to: str
    value: float
    asset: str
    category: str
    raw_contract_value: str
    raw_contract_address: str
    raw_contract_decimal: str
    block_timestamp: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetTransfer':
        try:
            # Extract nested data
            raw_contract = data.get('rawContract', {})
            metadata = data.get('metadata', {})
            
            return cls(
                blockNum=int(data['blockNum'], 16),
                uniqueId=data['uniqueId'],
                hash=data['hash'],
                from_address=data['from'],
                to=data['to'],
                value=float(data['value']),
                asset=data['asset'],
                category=data['category'],
                raw_contract_value=raw_contract.get('value'),
                raw_contract_address=raw_contract.get('address'),
                raw_contract_decimal=raw_contract.get('decimal'),
                block_timestamp=metadata.get('blockTimestamp')
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in asset transfer data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in asset transfer: {e}")

@dataclass
class SwapEvent:
    sender: str
    recipient: str
    amount0: int
    amount1: int
    sqrtPriceX96: int
    liquidity: int
    tick: int
    event: str
    logIndex: int
    transactionIndex: int
    transactionHash: str
    address: str
    blockHash: str
    blockNumber: int

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            args = attr_dict['args']
            return cls(
                sender=args['sender'],
                recipient=args['recipient'],
                amount0=args['amount0'],
                amount1=args['amount1'],
                sqrtPriceX96=args['sqrtPriceX96'],
                liquidity=args['liquidity'],
                tick=args['tick'],
                event=attr_dict['event'],
                logIndex=attr_dict['logIndex'],
                transactionIndex=attr_dict['transactionIndex'],
                transactionHash= "0x" + attr_dict['transactionHash'].hex() if isinstance(attr_dict['transactionHash'], bytes) else attr_dict['transactionHash'],
                address=attr_dict['address'],
                blockHash= "0x" + attr_dict['blockHash'].hex() if isinstance(attr_dict['blockHash'], bytes) else attr_dict['blockHash'],
                blockNumber=attr_dict['blockNumber']
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in SwapEvent data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in SwapEvent: {e}")

@dataclass
class SwapEventWithFee(SwapEvent):
    fee: int

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            base_event = SwapEvent.from_dict(attr_dict)
            fee = attr_dict.get('fee')
            if fee is None:
                raise ValueError("Missing 'fee' in SwapEventWithFee data")
            return cls(
                **base_event.__dict__,
                fee=fee
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in SwapEventWithFee data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in SwapEventWithFee: {e}")

@dataclass
class SwapEventV2:
    sender: str
    to: str
    amount0In: int
    amount1In: int
    amount0Out: int
    amount1Out: int
    event: str
    logIndex: int
    transactionIndex: int
    transactionHash: str
    address: str
    blockHash: str
    blockNumber: int

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            args = attr_dict['args']
            return cls(
                sender=args['sender'],
                to=args['to'],
                amount0In=args['amount0In'],
                amount1In=args['amount1In'],
                amount0Out=args['amount0Out'],
                amount1Out=args['amount1Out'],
                event=attr_dict['event'],
                logIndex=attr_dict['logIndex'],
                transactionIndex=attr_dict['transactionIndex'],
                transactionHash="0x" + attr_dict['transactionHash'].hex() if isinstance(attr_dict['transactionHash'], bytes) else attr_dict['transactionHash'],
                address=attr_dict['address'],
                blockHash="0x" + attr_dict['blockHash'].hex() if isinstance(attr_dict['blockHash'], bytes) else attr_dict['blockHash'],
                blockNumber=attr_dict['blockNumber']
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in SwapEventV2 data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in SwapEventV2: {e}")

@dataclass
class MintEvent:
    sender: str
    owner: str
    tickLower: int
    tickUpper: int
    amount: int
    amount0: int
    amount1: int
    event: str
    logIndex: int
    transactionIndex: int
    transactionHash: str
    address: str
    blockHash: str
    blockNumber: int

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            args = attr_dict['args']
            return cls(
                sender=args['sender'],
                owner=args['owner'],
                tickLower=args['tickLower'],
                tickUpper=args['tickUpper'],
                amount=int(args['amount']),
                amount0=int(args['amount0']),
                amount1=int(args['amount1']),
                event=attr_dict['event'],
                logIndex=attr_dict['logIndex'],
                transactionIndex=attr_dict['transactionIndex'],
                transactionHash="0x" + attr_dict['transactionHash'].hex() if isinstance(attr_dict['transactionHash'], bytes) else attr_dict['transactionHash'],
                address=attr_dict['address'],
                blockHash="0x" + attr_dict['blockHash'].hex() if isinstance(attr_dict['blockHash'], bytes) else attr_dict['blockHash'],
                blockNumber=attr_dict['blockNumber']
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in MintEvent data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in MintEvent: {e}")

@dataclass
class BurnEvent:
    owner: str
    tickLower: int
    tickUpper: int
    amount: int
    amount0: int
    amount1: int
    event: str
    logIndex: int
    transactionIndex: int
    transactionHash: str
    address: str
    blockHash: str
    blockNumber: int

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            args = attr_dict['args']
            return cls(
                owner=args['owner'],
                tickLower=args['tickLower'],
                tickUpper=args['tickUpper'],
                amount=int(args['amount']),
                amount0=int(args['amount0']),
                amount1=int(args['amount1']),
                event=attr_dict['event'],
                logIndex=attr_dict['logIndex'],
                transactionIndex=attr_dict['transactionIndex'],
                transactionHash="0x" + attr_dict['transactionHash'].hex() if isinstance(attr_dict['transactionHash'], bytes) else attr_dict['transactionHash'],
                address=attr_dict['address'],
                blockHash="0x" + attr_dict['blockHash'].hex() if isinstance(attr_dict['blockHash'], bytes) else attr_dict['blockHash'],
                blockNumber=attr_dict['blockNumber']
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in BurnEvent data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in BurnEvent: {e}")

@dataclass
class Log:
    blockHash: str
    address: str
    logIndex: int
    data: str
    removed: bool
    topics: list[str]
    blockNumber: int
    transactionIndex: int
    transactionHash: str

    @classmethod
    def from_dict(cls, attr_dict):
        try:
            return cls(
                blockHash="0x" + attr_dict['blockHash'].hex() if isinstance(attr_dict['blockHash'], bytes) else attr_dict['blockHash'],
                address=attr_dict['address'],
                logIndex=attr_dict['logIndex'],
                data="0x" + attr_dict['data'].hex() if isinstance(attr_dict['data'], bytes) else attr_dict['data'],
                removed=attr_dict['removed'],
                topics=[("0x" + topic.hex()) if isinstance(topic, bytes) else topic for topic in attr_dict['topics']],
                blockNumber=attr_dict['blockNumber'],
                transactionIndex=attr_dict['transactionIndex'],
                transactionHash="0x" + attr_dict['transactionHash'].hex() if isinstance(attr_dict['transactionHash'], bytes) else attr_dict['transactionHash']
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in FeesCollectedEvent data: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in FeesCollectedEvent: {e}")



def save_to_csv(logs, filename, cls):
    with open(filename, 'w') as f:
        writer = DataclassWriter(f, logs, cls)
        writer.write()
        print("Total entries", len(logs))

def load_from_csv(filename, cls):
    with open(filename, 'r') as f:
        reader = DataclassReader(f, cls)
        logs = []
        for row in reader:
            logs.append(row)
        return logs

