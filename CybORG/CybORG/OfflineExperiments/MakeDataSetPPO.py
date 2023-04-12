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

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

for b in [75, 150, 225]:
    ITERS = b
    config = (
        PPOConfig()
        #Each rollout worker uses a single cpu
        .rollouts(num_rollout_workers=NUM_WORKER, num_envs_per_worker=1, horizon=100)\
        .training(train_batch_size=BATCH_SIZE, gamma=0.9, lr=0.0001, 
                #   model={"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh",})\
                    model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh",})\
        .environment(disable_env_checking=True, env = 'CybORG')\
        # .resources(num_gpus=0)\
        .framework('tf')\
        .exploration(explore=True, exploration_config={"type": "RE3", "embeds_dim": 64, "beta_schedule": "constant", "sub_exploration": {"type": "StochasticSampling",},})\
        .offline_data(output=f"logs/PPO/{RED_AGENT}_no_decoy_{ITERS*BATCH_SIZE}", output_compress_columns=['prev_actions', 'prev_rewards', 'dones', 't', 'action_prob', 'action_logp', 'action_dist_inputs', 'advantages', 'value_targets'], #'eps_id', 'unroll_id', 'agent_index',
                    output_config={"format": "json"},)\
    )
    trainer = config.build() #use_copy=True

    def print_results(results_dict):
        train_iter = results_dict["training_iteration"]
        r_mean = results_dict["episode_reward_mean"]
        r_max = results_dict["episode_reward_max"]
        r_min = results_dict["episode_reward_min"]
        print(f"{train_iter:4d} \tr_mean: {r_mean:.1f} \tr_max: {r_max:.1f} \tr_min: {r_min: .1f}")

    for i in range(ITERS):
        print_results(trainer.train())