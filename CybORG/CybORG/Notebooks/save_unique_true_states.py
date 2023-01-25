from ray.rllib.offline.json_reader import JsonReader
import numpy as np
import numpy_indexed as npi
import pandas as pd

def convert_to_dataframe(np_true_state):
    column_dict = {}
    for node in range(13):
        column_dict[f"{node}_unknown"] = np_true_state[:,node*6+0]
        column_dict[f"{node}_known"] = np_true_state[:,node*6+1]
        column_dict[f"{node}_scanned"] = np_true_state[:,node*6+2]
        column_dict[f"{node}_none"] = np_true_state[:,node*6+3]
        column_dict[f"{node}_user"] = np_true_state[:,node*6+4]
        column_dict[f"{node}_privileged"] = np_true_state[:,node*6+5]
    
    dataset = pd.DataFrame(column_dict)
    return dataset

run_name = "TrueStates_200_4000_Meander_badblue"
input_reader = JsonReader(f"logs/APPO/{run_name}")


dfs = []
for e in range(800):
    print(f"loading {e}")
    data = input_reader.next()
#     print(data["pre_action_true_states"].shape)
#     print(data["obs"].shape)
#     print(data["pre_action_true_states"].shape)
    assert(data["obs"].shape[0] == data["pre_action_true_states"].shape[0])
    dfs.append(convert_to_dataframe(data["pre_action_true_states"]).drop_duplicates())
    dfs.append(convert_to_dataframe(data["blue_action_true_states"]).drop_duplicates())
    dfs.append(convert_to_dataframe(data["red_action_true_states"]).drop_duplicates())

print(f"concatenating")
dataset = pd.concat(dfs, ignore_index=True)

print(dataset.tail())

print(f"number of rows = {dataset.shape[0]}")
print("dropping duplicates...")

dataset = dataset.drop_duplicates()
print(f"number of rows = {dataset.shape[0]}")

import os  
os.makedirs('csv_data', exist_ok=True)
dataset.to_csv(f"csv_data/{run_name}.csv")