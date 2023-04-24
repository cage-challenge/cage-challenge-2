import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from multiprocessing.dummy import Pool
import gym
import tensorflow as tf 
#tf.compat.v1.enable_eager_execution()
from gym import error, spaces, utils
from gym.utils import seeding
from lstm_node_tranistion_model_feedback2 import CAGENodeTranistionModelLSTMFeedback2
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

sequence_length = 10
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
        self.reward_model.load('reward_model_lstm')

       
        self.state_tranistion_model = CAGENodeTranistionModelLSTMFeedback2()
        self.state_tranistion_model.load('NodeTranistionModelFA')
        self.state_tranistion_model.load_comp('NodeTranistionModelFC')
        self.init_state = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                                    0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                    1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                                    1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])
        self.states = np.zeros((sequence_length, state_len))
        self.states[0,:] = self.init_state
        self.actions_seq = np.zeros(sequence_length)
        self.clf = joblib.load('state_novelty.pkl')
        self.actions_onehot = np.zeros((sequence_length, 41))
        self.time_step = 0

    def step(self, action):

        if self.time_step < sequence_length:
            self.actions_seq[self.time_step] = action
        else:
            self.actions_seq = np.roll(self.actions_seq,-1,axis=0)
            self.actions_seq[-1] = action
            self.actions_onehot = np.roll(self.actions_onehot,-1,axis=0)
            self.actions_onehot[-1,action] = 1
            
        self.time_step+=1
     
        #flag = 5
        #while flag > 0:
        state = self.state_tranistion_model.forward(self.states, self.actions_seq)
        #    if any((state==unique_states).all(1)):
        #        flag = 0
        #    flag -= 1

        state_action = np.concatenate([self.states, self.actions_onehot], axis=-1)

        reward = self.reward_model.forward(np.array([state_action]), np.array([state]))
        if self.time_step < 5:
            self.states[self.time_step] = state
        else:
            self.states = np.roll(self.states,-1,axis=0)
            self.states[-1] = state

        self.step_count += 1
        done = self.step_count == 99
        if done:
            self.step_count = 0
        return state, reward, done, {'entropy': self.reward_model.get_entropy()}

    def reset(self):
        self.time_step = 0
        self.states = np.zeros((sequence_length, state_len))
        self.states[0,:] = self.init_state
        self.actions_seq = np.zeros(sequence_length)        

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
OOD_r = []
ID = []
ID_States = []
ID_r = []
both = []

history = []
       
def run_episode(k):
    hist = np.zeros((100, 91))
    WM = WorldMovelEnv()
    s = WM.reset()
    done = False
    index = 0
    while not done:
        action = np.random.randint(41)
        hist[index,:] = s
        ns, r, done, i = WM.step(action)
        if any((s==unique_states).all(1)):
            ID_States.append(s)
        else:
            OOD_States.append(s)
        if any((ns==unique_states).all(1)):
            ID.append(i['entropy'])
            ID_r.append(r)
        else:
            OOD.append(i['entropy'])
            OOD_r.append(r)
        if not any((ns==unique_states).all(1)) and not any((s==unique_states).all(1)):
            both.append(i['entropy'])
        s = ns
        index += 1
    hist[index,:] = s
    history.append(hist)

#p = Pool(5)
#p.map(run_episode, range(10))
#p.close()
#p.join()

for i in trange(50):
    run_episode(i)

OOD = np.array(OOD)
ID = np.array(ID)

OOD_States = np.array(OOD_States)
ID_States = np.array(ID_States)
both = np.array(both)

print('ID mean: ', ID.mean())
print('ID std: ', ID.std())
print('ID shape: ', ID.shape)
print('ID reward: ', np.mean(np.array(ID_r)))
print('OOD mean: ', OOD.mean())
print('OOD std: ', OOD.std())
print('OOD shape: ', OOD.shape)
print('OOD reward: ', np.mean(np.array(OOD_r)))
print('Both mean: ', both.mean())
print('Both std: ', both.std())
print('Both shape: ', both.shape)

print(ID.shape[0]/(ID.shape[0]+OOD.shape[0]))

np.save('random_walks_dream.npy', np.array(history))

np.save('ID.npy', ID)
np.save('OOD.npy', OOD)

np.save('ID_States.npy', ID_States)
np.save('OOD_States.npy', OOD_States)

