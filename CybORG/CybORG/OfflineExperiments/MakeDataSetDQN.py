import inspect
import time
from statistics import mean, stdev
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

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

warnings.filterwarnings('ignore')

# id = -1
# def next_id():
#     global id
#     id += 1
#     return id

NUM_WORKER = 4
BATCH_SIZE = 2000
ITERS = 100
RED_AGENT = "B_Line"
#RED_AGENT = "Meander"

def env_creator(env_config: dict):
    # import pdb; pdb.set_trace()
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2_No_Decoy.yaml'
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

register_env(name="CybORG", env_creator=env_creator)

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray import air, tune

for b in [100000, 200000, 300000]:

    tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop={"timesteps_total": b},
            local_dir='results/DQN_'+str(b)+'_random', name="tune",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=500, 
            ),
        ),
        param_space={
            # CC3 specific.
            "env": "CybORG",
            # General
            "num_gpus": 1,
            "num_workers": 3,
            "horizon": 100,
            "num_envs_per_worker": 4,
            "n_step":  10,
            #algo params
            "train_batch_size": 8,
            "lr": 0.0000005,
            "gamma": 0.9,
            "framework": 'tf',
            "model": {
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                },
            "num_atoms": 50,
            "v_min": -100.0,
            "v_max": 0.0,
            "noisy": True,       
            "output": f"logs/DQN/{RED_AGENT}_no_decoy_{b}_random",
            "output_config": {"format": "json"},
            "explore_config": 
             {  
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.5,
                "final_epsilon": 1.1,
                "epsilone_timesteps": 5000000,
             }
         
        },
    ).fit()

        