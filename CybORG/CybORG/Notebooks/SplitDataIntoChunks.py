import pandas as pd
import os

data_path = "csv_data/TrueStatesObsActsRwds_1221_4000_B_Line"
data_ext = ".parquet"

chunk_size = 500_000




new_data_path = f"{data_path}_chunks_{chunk_size}"
os.makedirs(new_data_path, exist_ok=True)

df=pd.read_parquet(data_path+data_ext)

for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
    chunk_end = chunk_start + chunk_size
    if chunk_end > len(df):
        chunk_end = len(df)
    new_chunk_path = f"{new_data_path}/chunk_{i}.parquet"
    print(f"saving chunk {i} : {chunk_start} - {chunk_end} to {new_chunk_path}")
    df.iloc[chunk_start:chunk_end].to_parquet(new_chunk_path)
