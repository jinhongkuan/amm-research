from dataclasses import dataclass
from typing import Dict, Any
from decimal import Decimal
from dataclass_csv import DataclassWriter, DataclassReader

@dataclass
class AssetTransfer:
    blockNum: str
    uniqueId: str
    hash: str
    from_address: str
    to: str
    value: Decimal
    asset: str
    category: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetTransfer':
        try:
            return cls(
                blockNum=data['blockNum'],
                uniqueId=data['uniqueId'],
                hash=data['hash'],
                from_address=data['from'],
                to=data['to'],
                value=Decimal(str(data['value'])),
                asset=data['asset'],
                category=data['category'],
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

def load_from_csv(filename, cls):
    with open(filename, 'r') as f:
        reader = DataclassReader(f, cls)
        logs = []
        for row in reader:
            logs.append(row)
        return logs

