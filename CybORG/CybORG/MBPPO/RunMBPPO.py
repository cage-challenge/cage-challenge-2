import inspect
import time
from statistics import mean, stdev
import sys 
sys.path.append('../')
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent, GreenAgent, RedMeanderAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
import os

from ray.tune.registry import register_env
from CybORG.Agents.Wrappers.rllib_wrapper import RLlibWrapper
import warnings
import numpy as np
from ray import air, tune
import ray

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#ray.init(log_to_driver=False)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.autograph.set_verbosity(0)


NUM_WORKER = 20
BATCH_SIZE = 4000
ITERS = 80
RED_AGENT = "Meander"
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
#for gpu in gpus:
#        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15360)])

def env_creator(env_config: dict):
    # import pdb; pdb.set_trace()
    path = '../Shared/Scenarios/Scenario2_No_Decoy.yaml'
    #path = str(inspect.getfile(CybORG))
    #path = path[:-10] + '/Shared/Scenarios/Scenario2_No_Decoy.yaml'
    if RED_AGENT == "B_Line":
        agents = {"Red": B_lineAgent, "Green": GreenAgent}
    else:
        agents = {"Red": RedMeanderAgent, "Green": GreenAgent}

    cyborg = CybORG(scenario_file=path, environment='sim', agents=agents)
    env = RLlibWrapper(env=cyborg, agent_name="Blue", max_steps=100)
    return env

def print_results(results_dict):
    train_iter = results_dict["training_iteration"]
    r_mean = results_dict["episode_reward_mean"]
    r_max = results_dict["episode_reward_max"]
    r_min = results_dict["episode_reward_min"]
    print(f"{train_iter:4d} \tr_mean: {r_mean:.1f} \tr_max: {r_max:.1f} \tr_min: {r_min: .1f}")
    return r_mean

register_env(name="CybORG", env_creator=env_creator)

from MBPPO import MBPPOConfig

for s in range(0,5):
    config = (
        MBPPOConfig()
        #Each rollout worker uses a single cpu
        .rollouts(num_rollout_workers=NUM_WORKER, num_envs_per_worker=1)\
        .training(train_batch_size=BATCH_SIZE, gamma=0.9, lr=0.0001, seq_len=80,
                #   model={"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh",})\
                    model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh",})\
                          # "use_lstm": False,
                           # "max_seq_len": 5,
                          #  "lstm_cell_size": 256,})\
        .environment(disable_env_checking=True, env = 'CybORG')\
        .framework('tf2')\
        .resources(num_gpus=0)
        #.exploration(explore=True, exploration_config={"type": "RE3", "embeds_dim": 128, "beta_schedule": "constant", "sub_exploration": {"type": "StochasticSampling",},})\
    )
    trainer = config.build() 

    import warnings
    warnings.filterwarnings("ignore")

    rewards = np.zeros(ITERS)
    for i in range(0,ITERS):
        rewards[i] = print_results(trainer.train())
        np.save('mbppo_80_meander_8000'+str(s), rewards)