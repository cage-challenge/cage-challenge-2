from ray.rllib.offline.json_reader import JsonReader
import numpy as np
import pandas as pd
from tqdm import trange
import os

def node_action(action, node):
    vec = np.zeros(3)
    if action < 2: 
        return vec
    action -= 2
    if action % 13 == node:
        #Analyse #Remove #Resotre
        vec[int(action) // 13] = 1
    return vec
    
data_path = '/home/ubuntu/u75a-Data-Efficient-Decisions/CybORG/CybORG/OfflineExperiments/logs/PPO/B_Line_no_decoy_300000'
input_reader = JsonReader(data_path)
num_episodes = int(300000/100)
num_data_points = num_episodes * 99
state_length = 91
num_actions = 41
seq_len = 10

states = np.zeros((num_data_points, seq_len, state_length))
rewards = np.zeros(num_data_points)
next_states = np.zeros((num_data_points, state_length))
nodes = np.zeros((num_data_points*13, seq_len, 7))
actions = np.zeros((num_data_points*13, seq_len, 3))
node_id = np.zeros((num_data_points*13, seq_len, 13))
next_nodes = np.zeros((num_data_points*13, 7))
exploit = np.zeros((num_data_points*13, seq_len, 13))
scan = np.zeros((num_data_points*13, seq_len, 13))
privileged = np.zeros((num_data_points*13, seq_len, 13))
user = np.zeros((num_data_points*13, seq_len, 13))
unknown = np.zeros((num_data_points*13, seq_len, 13))
no  = np.zeros((num_data_points*13, seq_len, 13))

s_index = 0
index = 0
for i in trange(int(num_episodes/5)):
    samples = input_reader.next()
    ts = 0
    for i in range(samples['dones'].shape[0]-1):
        if samples['dones'][i]:
            ts = 0
            continue
        steps = ts if ts < seq_len else seq_len-1  
        states[s_index,:steps+1,:] =  samples['obs'][i-steps:i+1,:]
        next_states[s_index,:] = samples['obs'][i+1,:]
        rewards[s_index] = samples['rewards'][i]
        s_index += 1
        for n in range(13):     
            encoding = np.zeros((steps, 13))
            encoding[:steps+1,n] = 1
            node_id[index,:steps+1,n] = 1
            nodes[index,:steps+1,:] = samples['obs'][i-steps:i+1,int(n*7):int(n*7)+7]
            actions[index,:steps+1,:] = np.array([node_action(samples['actions'][i], n) for i in range(i-steps,i+1)])
            next_nodes[index,:] = samples['obs'][i+1,int(n*7):int(n*7)+7]
            exploit[index,:steps+1,:] = samples['obs'][i-steps:i+1,np.arange(0,91,step=7)]
            scan[index,:steps+1,:] = samples['obs'][i-steps:i+1,np.arange(1,91,step=7)]
            privileged[index,:steps+1,:] = samples['obs'][i-steps:i+1,np.arange(3,91,step=7)]
            user[index,:steps+1,:] = samples['obs'][i-steps:i+1,np.arange(4,91,step=7)]
            unknown[index,:steps+1,:] = samples['obs'][i-steps:i+1,np.arange(5,91,step=7)]
            no[index,:steps+1,:]  = samples['obs'][i-steps:i+1,np.arange(6,91,step=7)]
            index += 1
        ts += 1
    
data_path = data_path + '/data_seqence_10'
os.mkdir(data_path)
np.save(data_path + '/nodes.npy', nodes)
np.save(data_path + '/actions.npy', actions)
np.save(data_path + '/node_id.npy', node_id)
np.save(data_path + '/next_nodes.npy', next_nodes)
np.save(data_path + '/exploit.npy', exploit)
np.save(data_path + '/scan.npy', scan)
np.save(data_path + '/privileged.npy', privileged)
np.save(data_path + '/user.npy', user)
np.save(data_path + '/unknown.npy', unknown)
np.save(data_path + '/no.npy', no)