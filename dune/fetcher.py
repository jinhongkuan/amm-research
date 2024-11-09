import os
import pandas as pd
from dune_client.client import DuneClient

api_key = os.getenv("DUNE_API_KEY")
query_id = os.getenv("DUNE_QUERY_ID")
if api_key is None:
    raise ValueError("DUNE_API_KEY environment variable is not set.")
if query_id is None:
    raise ValueError("DUNE_QUERY_ID environment variable is not set.")
dune = DuneClient(api_key)
df = dune.get_latest_result_dataframe(query_id)
df.to_csv("dune_data.csv", index=False)