import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import gym
import tensorflow as tf 
#tf.compat.v1.enable_eager_execution()
from gym import error, spaces, utils
from gym.utils import seeding
from CybORG.MBPPO.lstm_node_tranistion_model import CAGENodeTranistionModelLSTM
from CybORG.MBPPO.reward_model import CAGERewardModel
from tensorflow.keras.models import Model
import numpy as np
from tqdm import trange
import joblib

import inspect
import time
from statistics import mean, stdev
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent, GreenAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.rllib_wrapper import RLlibWrapper
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray
warnings.filterwarnings('ignore')


state_len = 91
num_actions = 41
encoding_len = state_len + num_actions
NUM_NODES = 13
NODE_CLASSES = [3, 4]

REWARD_MODEL = ''
STATE_TRANISION_MODEL = ''
SEQ_LEN = 10

class WorldMovelEnv(gym.Env):

    def __init__(self):

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(state_len,))
        self.action_space = gym.spaces.Discrete(num_actions)

        self.step_count = 0

        # Reward Model
        self.reward_model = CAGERewardModel()
        self.reward_model.load(REWARD_MODEL)

       
        self.state_tranistion_model = CAGENodeTranistionModelLSTM()
        self.state_tranistion_model.load(STATE_TRANISION_MODEL)
        self.init_state = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                                    0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                    1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                                    1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])
        self.states = np.zeros((SEQ_LEN, state_len))
        self.states[0,:] = self.init_state
        self.actions_seq = np.zeros(SEQ_LEN)
        self.clf = joblib.load('state_novelty.pkl')

    def step(self, action):

        self.actions_seq[-1] = action
        self.actions_seq = np.roll(self.actions_seq,1,axis=0)
        valid = -1; state = None
        state = self.state_tranistion_model.forward(self.states, self.actions_seq)

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


def env_creator_wm(config):
    return WorldMovelEnv()

def env_creator_cyborg(env_config: dict):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    agents = {"Red": B_lineAgent, "Green": GreenAgent}
    cyborg = CybORG(scenario_file=path, environment='sim', agents=agents)
    env = RLlibWrapper(env=cyborg, agent_name="Blue", max_steps=100)
    return env

register_env(name="CybORG_WM", env_creator=env_creator_wm)

config = (
        PPOConfig()
        .rollouts(num_rollout_workers=10, num_envs_per_worker=1)\
        .training(train_batch_size=2000, gamma=0.9, lr=0.0001, 
                    model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh",})
        .environment(disable_env_checking=True, env ='CybORG_WM')\
        .framework('tf2')\
        .resources(num_gpus=1)
    )
trainer = config.build()