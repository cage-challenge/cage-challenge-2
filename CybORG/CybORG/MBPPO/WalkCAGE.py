import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import gym
import tensorflow as tf 
#tf.compat.v1.enable_eager_execution()
from gym import error, spaces, utils
from gym.utils import seeding
from CybORG.MBPPO.lstm_node_tranistion_model_feedback2 import CAGENodeTranistionModelLSTMFeedback2
from CybORG.MBPPO.reward_model import CAGERewardModel
from tensorflow.keras.models import Model
import numpy as np
from tqdm import trange
import joblib
from multiprocessing.dummy import Pool

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

def env_creator_cyborg(env_config: dict):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    agents = {"Red": B_lineAgent, "Green": GreenAgent}
    cyborg = CybORG(scenario_file=path, environment='sim', agents=agents)
    env = RLlibWrapper(env=cyborg, agent_name="Blue", max_steps=100)
    return env

history = []
rewards = []
def run_episode(i):
    cyborg = env_creator_cyborg({})
    hist = np.zeros((100, 91))
    r = 0
    observation = cyborg.reset()
    for j in range(100):
        hist[j,:] = observation
        action = np.random.randint(42)
        observation, rew, done, info = cyborg.step(action)
        r += rew
    hist[-1,:] = observation
    rewards.append(r)
    history.append(hist)
p = Pool(10)
p.map(run_episode, range(100))
p.close()
p.join()
print(rewards)

np.save('random_walks_real.npy', np.array(history))
