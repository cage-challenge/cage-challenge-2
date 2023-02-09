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

NUM_WORKER = 10
BATCH_SIZE = 1000
ITERS = 200
# RED_AGENT = "B_Line"
RED_AGENT = "Meander"

class CustomTrueStateCallbackSaver(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        # self.id = next_id()
        # print(f"INIT A CALLBACKS {self.id}")
        self.worker_to_pres = {}
        self.worker_to_blues = {}
        self.worker_to_reds = {}
        self.worker_to_afterstates = {}

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        
        env = base_env.get_sub_environments()
        true_state_sequences = env[env_index].env.env.env.env.env.environment_controller.pop_additional_states_sequences()
        # print(f"worker = {worker}")
        # print(len(true_state_sequences),flush=True)
        # print(len(true_state_sequences[0]),flush=True)
        if len(true_state_sequences[0]) > 0:
            # print(true_state_sequences[0][30],flush=True)
            # print(true_state_sequences[1][30],flush=True)
            # print(true_state_sequences[2][30],flush=True)

            pres_np  = np.array(true_state_sequences[0])
            blues_np  = np.array(true_state_sequences[1])
            reds_np  = np.array(true_state_sequences[2])
            afterstates = np.array(true_state_sequences[3])

            if len(true_state_sequences[0]) > 1:
                red1_eq_pres0 = reds_np[:-1,:]==pres_np[1:,:]
                
                assert np.alltrue(red1_eq_pres0), "failed assumption that no true state change between post-red action and next pre-action states"
                
                # TODO: move these assers to end sample to test over the full batch
                # assert not np.all(pres_np==blues_np), "failed assumption that true state can change after blue and before red actions"
                assert not np.all(blues_np==reds_np), "failed assumption that true state can change after blue and red actions"

            if worker not in self.worker_to_pres:
                self.reset_worker_stores(worker)

            self.worker_to_pres[worker].append(pres_np) # np.concatenate(self.worker_to_pres[worker], pres_np)
            self.worker_to_blues[worker].append(blues_np) # = np.concatenate(self.worker_to_blues[worker], blues_np)
            self.worker_to_reds[worker].append(reds_np) # np.concatenate(self.worker_to_reds[worker], reds_np)
            self.worker_to_afterstates[worker].append(afterstates)

    def reset_worker_stores(self, worker):
        self.worker_to_pres[worker] = [] #np.array([[]],dtype=np.int64)
        self.worker_to_blues[worker] = [] # np.array([[]],dtype=np.int64)
        self.worker_to_reds[worker] = [] #np.array([[]],dtype=np.int64)
        self.worker_to_afterstates[worker] = []
    
    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        """Called at the end RolloutWorker.sample().

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            samples (SampleBatch): Batch to be returned. You can mutate this
                object to modify the samples generated.
            kwargs: Forward compatibility placeholder.
        """

        # print(f"worker = {worker}")
        samples_len = len(samples['obs'])
        #print(f"SAMPLES LEN = {samples_len}, is {len(samples['obs'])/100} sequences of 100, truncating true state arrays if they do not match")

        # Concat all runs into a single experience batch
        sample_pres = np.concatenate(self.worker_to_pres[worker])
        sample_blues = np.concatenate(self.worker_to_blues[worker])
        sample_reds = np.concatenate(self.worker_to_reds[worker])
        sample_afterstate = np.concatenate(self.worker_to_afterstates[worker])

        # Truncate and save to the SampleBatch dict
        samples["pre_action_true_states"] = sample_pres[:samples_len,:]
        samples["blue_action_true_states"] = sample_blues[:samples_len,:]
        samples["red_action_true_states"] = sample_reds[:samples_len,:]
        samples["afterstates"] = sample_afterstate[:samples_len,:]

       # assert not np.all(sample_pres==sample_blues), "failed assumption that true state can change after blue and before red actions"
      #  assert not np.all(sample_blues==sample_reds), "failed assumption that true state can change after blue and red actions"

        self.reset_worker_stores(worker)
        # samples["yessss"]= np.array([len(samples['obs'])])
            
def env_creator(env_config: dict):
    # import pdb; pdb.set_trace()
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2Small.yaml'
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


# TODO: maybe add horizon to the callback initialiser
config = (
    PPOConfig()
    #Each rollout worker uses a single cpu
    .rollouts(num_rollout_workers=NUM_WORKER, num_envs_per_worker=1, horizon=100)\
    .training(train_batch_size=BATCH_SIZE, gamma=0.99, lr=0.00005, 
            #   model={"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh",})\
                model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh",})\
    .environment(disable_env_checking=True, env = 'CybORG')\
    # .resources(num_gpus=0)\
    .framework('tf')\
    # .exploration(explore=True, exploration_config={"type": "RE3", "embeds_dim": 128, "beta_schedule": "constant", "sub_exploration": {"type": "StochasticSampling",},})\
    .exploration(explore=True, exploration_config={"type": "RE3", "embeds_dim": 64, "beta_schedule": "constant", "sub_exploration": {"type": "StochasticSampling",},})\
    .offline_data(output=f"logs/APPO/TrueStates_{ITERS}_{BATCH_SIZE}_{RED_AGENT}_small_4_bit", output_compress_columns=['prev_actions', 'prev_rewards', 'dones', 't', 'action_prob', 'action_logp', 'action_dist_inputs', 'advantages', 'value_targets'], #'eps_id', 'unroll_id', 'agent_index',
                 output_config={"format": "json"},)\
    .callbacks(CustomTrueStateCallbackSaver)
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