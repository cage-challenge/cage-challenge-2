import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import gym
import tensorflow as tf 
#tf.compat.v1.enable_eager_execution()
from gym import error, spaces, utils
from gym.utils import seeding
from lstm_node_tranistion_model import CAGENodeTranistionModelLSTM
from reward_model import CAGERewardModel
from tensorflow.keras.models import Model
import numpy as np
from tqdm import trange
import joblib

#Get unique states 
data_path = '/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/Notebooks/logs/PPO/no_decoy_200000'
state = np.load(data_path + '/data/state.npy')
unique_states = np.unique(state, axis=0)
unique_states = list(unique_states)

init_state = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                                    0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                    1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                                    1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])

any((init_state==unique_states).all(1))

sequence_length = 3
state_len = 91
num_actions = 41
encoding_len = state_len + num_actions
NUM_NODES = 13
NODE_CLASSES = [3, 4]

class WorldMovelEnv(gym.Env):

    def __init__(self):

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(state_len,))
        self.action_space = gym.spaces.Discrete(num_actions)

        self.step_count = 0

        # Reward Model
        self.reward_model = CAGERewardModel()
        self.reward_model.load('reward_model')

       
        self.state_tranistion_model = CAGENodeTranistionModelLSTM()
        self.state_tranistion_model.load('NodeTranistionModel')
        self.init_state = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                                    0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                    1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                                    1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])
        self.states = np.zeros((10, state_len))
        self.states[0,:] = self.init_state
        self.actions_seq = np.zeros(10)
        self.clf = joblib.load('state_novelty.pkl')

    def step(self, action):

        self.actions_seq[-1] = action
        self.actions_seq = np.roll(self.actions_seq,1,axis=0)
        valid = -1; state = None
        #while valid == -1:
        state = self.state_tranistion_model.forward(self.states, self.actions_seq)
           # valid = self.clf.predict(np.expand_dims(state, axis=0))[0]
        self.states[-1,:] = state

        reward = self.reward_model.forward(np.array([np.concatenate([self.states[0,:], self.states[-1,:]])]))
        self.states = np.roll(self.states,1,axis=0)
        self.step_count += 1
        done = self.step_count == 99
        if done:
            self.step_count = 0
        return state, reward, done, {'entropy': self.reward_model.get_entropy()}

    def reset(self):
        step_count = 0
        self.states = np.zeros((10, state_len))
        self.states[0,:] = self.init_state
        self.actions_seq = np.zeros(10)        

        return self.init_state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


def env_creator(config):
    return WorldMovelEnv()


WM = WorldMovelEnv()

OOD = []
OOD_States = []
ID = []
ID_States = []
both = []
for i in trange(100):
    done = False
    s = WM.reset()
    while not done:
        action = np.random.randint(41)
        ns, r, done, i = WM.step(action)
        if any((s==unique_states).all(1)):
            ID_States.append(s)
        else:
            OOD_States.append(s)
        if any((ns==unique_states).all(1)):
            ID.append(i['entropy'])
        else:
            OOD.append(i['entropy'])
        if not any((ns==unique_states).all(1)) and not any((s==unique_states).all(1)):
            both.append(i['entropy'])
        s = ns
        

OOD = np.array(OOD)
ID = np.array(ID)

OOD_States = np.array(OOD_States)
ID_States = np.array(ID_States)
both = np.array(both)

print('ID mean: ', ID.mean())
print('ID std: ', ID.std())
print('ID shape: ', ID.shape)
print('OOD mean: ', OOD.mean())
print('OOD std: ', OOD.std())
print('OOD shape: ', OOD.shape)
print('Both mean: ', both.mean())
print('Both std: ', both.std())
print('Both shape: ', both.shape)

np.save('ID.npy', ID)
np.save('OOD.npy', OOD)

np.save('ID_States.npy', ID_States)
np.save('OOD_States.npy', OOD_States)

