#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from ray.rllib.offline.json_reader import JsonReader
import numpy as np
import numpy_indexed as npi
import pandas as pd
from IPython.display import display
import graphviz
import numpy as np
import ipywidgets as widgets


# In[70]:


def convert_ts_np_to_df(df,np_true_state,prefix):
#     column_dict = {}
    newdf = pd.DataFrame()
    for node in range(13):
        newdf[f"{node}_ts_{prefix}_known_status"] = (np_true_state[:,node*6+0] + (np_true_state[:,node*6+1]*2) + (np_true_state[:,node*6+2]*3)) -1
        newdf[f"{node}_ts_{prefix}_access_status"] = (np_true_state[:,node*6+3] + (np_true_state[:,node*6+4]*2) + (np_true_state[:,node*6+5]*3)) -1  
        assert((newdf[f"{node}_ts_{prefix}_known_status"]>=0).all())
        assert((newdf[f"{node}_ts_{prefix}_known_status"]<3).all())
        assert((newdf[f"{node}_ts_{prefix}_access_status"]>=0).all())
        assert((newdf[f"{node}_ts_{prefix}_access_status"]<4).all())
        #     df = pd.DataFrame(column_dict)
    return pd.concat([df, newdf],axis=1)

def convert_obs_np_to_df(df,np_blue_obs):
#     column_dict = {}
    newdf = pd.DataFrame()
    for node in range(13):
        newdf[f"{node}_obs_blue_activity"] = (np_blue_obs[:,node*7+0] + (np_blue_obs[:,node*7+1]*2) + (np_blue_obs[:,node*7+2]*3)) -1
        newdf[f"{node}_obs_blue_compromised"] = (np_blue_obs[:,node*7+3] + (np_blue_obs[:,node*7+4]*2) + (np_blue_obs[:,node*7+5]*3) + (np_blue_obs[:,node*7+6]*4)) -1  
        
#         assert(df[f"{node}_obs_blue_activity"])
#         print(np_blue_obs[:,node*6+0:node*6+3])
        assert((newdf[f"{node}_obs_blue_activity"]>=0).all())
        assert((newdf[f"{node}_obs_blue_activity"]<3).all())
        assert((newdf[f"{node}_obs_blue_compromised"]>=0).all())
        assert((newdf[f"{node}_obs_blue_compromised"]<4).all())
#         print(np_blue_obs[:,node*6+3:node*6+7])
#         print(df[f"{node}_obs_blue_compromised"])
    return pd.concat([df, newdf],axis=1)

def convert_acts_rwds_nps_to_df(df, np_actions, np_rewards):
    newdf = pd.DataFrame()
    newdf["action_blue"] = np_actions
    newdf["reward"] = np_rewards
    return pd.concat([df, newdf],axis=1)


# In[6]:


def create_empty_df():
    columns = []
    for node in range(13):
        columns.append(f"{node}_ts_pre_known_status")
        columns.append(f"{node}_ts_pre_access_status")

        columns.append(f"{node}_ts_blue_known_status")
        columns.append(f"{node}_ts_blue_access_status")

        columns.append(f"{node}_ts_red_known_status")
        columns.append(f"{node}_ts_red_access_status")

        columns.append(f"{node}_obs_blue_activity")
        columns.append(f"{node}_obs_blue_compromised")

    columns.append(f"action_blue")
    columns.append(f"reward")


    full_df = pd.DataFrame(columns=columns)
    return full_df


# In[77]:


def convert_rllib_data_to_pandas(path):
    input_reader = JsonReader(path)

    # dfs = []
    full_df = create_empty_df()

    for e in range(4884):#4884
        data = input_reader.next()

        df = pd.DataFrame()

    #     print(data["actions"].shape) max 145?
    #     print(data.keys())
    #     print(data["rewards"])

    #     print(f"loading {e}")
        data = input_reader.next()
        assert(data["obs"].shape[0] == data["pre_action_true_states"].shape[0])

        df = convert_ts_np_to_df(df, data["pre_action_true_states"],"pre")
        df = convert_ts_np_to_df(df, data["blue_action_true_states"],"blue")
        df = convert_ts_np_to_df(df, data["red_action_true_states"],"red")

        df = convert_obs_np_to_df(df, data["obs"])
        df = convert_acts_rwds_nps_to_df(df, data["actions"], data["rewards"])

        full_df = pd.concat([full_df, df],axis=0)
        print(len(full_df))
    #     print(data["obs"][0])
    #     df[""]
    #     TODO: convert blue obs and action and concatenate
    return full_df
    


# In[79]:


# import os  
# run_name = "TrueStatesObsActsRwds_1221_4000_B_Line"
# os.makedirs('csv_data', exist_ok=True)
# full_df.to_csv(f"csv_data/{run_name}.csv")


# In[9]:


def read_df_in_chunks(file_path, columns=None):
    chunksize = 10 ** 6

    full_df = create_empty_df()

    if columns:
        full_df = full_df[columns]

    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        for index, chunk in enumerate(reader):
    #         ds2 = pd.read_csv()
            print(f"reading chunk {index}")
            chunk = chunk[columns] if columns else chunk
            full_df = pd.concat([full_df, chunk],axis=0)

    print(f"data frame size = {len(full_df)}\nColumn names are:")
    for col in full_df.columns:
        print(col)
    return full_df


# In[10]:

if __name__ == "__main__":
    # run_name = "TrueStatesObsActsRwds_1221_4000_B_Line"
    # os.makedirs('csv_data', exist_ok=True)
    # df = convert_rllib_data_to_pandas("logs/APPO/TrueStates_1221_4000_B_Line")
    read_df = read_df_in_chunks('csv_data/TrueStatesObsActsRwds_1221_4000_B_Line.csv')

