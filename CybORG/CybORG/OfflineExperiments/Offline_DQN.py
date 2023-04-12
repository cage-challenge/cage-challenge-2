import inspect
import time
from statistics import mean, stdev
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent, GreenAgent
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
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["SM_FRAMEWORK"] = "tf.keras"

RED_AGENT = "B_Line"

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

register_env(name="CybORG", env_creator=env_creator)

from ray import air, tune

tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop={"timesteps_total": 1e6},
            local_dir='results/DQN_offline/b_line100000', name="tune",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=500, 
            ),
        ),
        param_space={
            # CC3 specific.
            "env": "CybORG",
            # General
            "num_gpus": 1,
            "num_workers": 4,
            "horizon": 100,
            "num_envs_per_worker": 1,
            "n_step":  tune.grid_search([5]),
            #algo params
            "train_batch_size": tune.grid_search([1000]),
            "lr": 0.001,
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
            "input": '/home/ubuntu/u75a-Data-Efficient-Decisions/CybORG/CybORG/OfflineExperiments/logs/DQN/B_Line_no_decoy_100000',
            "evaluation_num_workers": 2,
            "evaluation_interval": 1,
            "evaluation_duration": 30,
            "evaluation_config": {"input": "sampler"}
        },
    ).fit()