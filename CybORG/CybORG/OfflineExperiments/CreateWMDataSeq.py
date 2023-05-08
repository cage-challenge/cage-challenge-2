from ray.rllib.offline.json_reader import JsonReader
import numpy as np
import pandas as pd
from tqdm import trange
import os

data_path = data_path = '/home/ubuntu/u75a-Data-Efficient-Decisions/CybORG/CybORG/OfflineExperiments/logs/PPO/B_Line_no_decoy_800000'
input_reader = JsonReader(data_path)
num_episodes = int(800000/100)
num_data_points = num_episodes * 99
state_length = 91
num_actions = 41
seq_len = 40

states = np.zeros((num_data_points, seq_len, state_length), dtype=np.int8)
rewards = np.zeros(num_data_points)
next_states = np.zeros((num_data_points, state_length), dtype=np.int8)
action_hist_oh = np.zeros((num_data_points, seq_len, num_actions), dtype=np.int8)
node_id = np.zeros((num_data_points*13, 13), dtype=np.int8)
next_nodes = np.zeros((num_data_points*13, 7), dtype=np.int8)

samples = input_reader.next()
episodes_per_sample = samples['dones'].shape[0] / 100

s_index = 0
index = 0
for i in trange(int(num_episodes/episodes_per_sample)):
    samples = input_reader.next()
    ts = 0
    for i in range(samples['dones'].shape[0]-1):
        if samples['dones'][i]:
            ts = 0
            continue
        steps = ts if ts < seq_len else seq_len-1 
        pad = seq_len-(ts+1) if ts < seq_len else 0  
        states[s_index,pad:,:] =  samples['obs'][i-steps:i+1,:]
        next_states[s_index,:] = samples['obs'][i+1,:]
        rewards[s_index] = samples['rewards'][i]
        action_hist_oh[s_index,pad:,:] = np.eye(41)[samples['actions'][i-steps:i+1]]
        s_index += 1
        for n in range(13):     
            node_id[index,n] = 1
            next_nodes[index,:] = samples['obs'][i+1,int(n*7):int(n*7)+7]
            index += 1
    
data_path = data_path + '/data_seqence_40'
#os.mkdir(data_path)
np.save(data_path + '/states.npy', states)
np.save(data_path + '/next_states.npy', next_states)
np.save(data_path + '/actions.npy', action_hist_oh)
np.save(data_path + '/node_id.npy', node_id)
np.save(data_path + '/next_nodes.npy', next_nodes)
np.save(data_path + '/rewards.npy', rewards)