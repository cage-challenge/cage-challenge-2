"""
Proximal Policy Optimization (PPO)
==================================

This file defines the distributed Algorithm class for proximal policy
optimization.
See `ppo_[tf|torch]_policy.py` for the definition of the policy loss.

Detailed documentation: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
"""

import logging
from typing import List, Optional, Type, Union
import numpy as np
from bisect import bisect_left

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit 
logger = logging.getLogger(__name__)


class MBPPOConfig(PGConfig):
    """Defines a configuration class from which a PPO Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> config = PPOConfig()  # doctest: +SKIP
        >>> config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = PPOConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)  # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(  # doctest: +SKIP
        ... lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
        ... )
        >>> # Set the config object's env.
        >>> config = config.environment(env="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "PPO",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or MBPPO)

        # fmt: off
        # __sphinx_doc_begin__
        # PPO specific settings:
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 0.99
        self.kl_coeff = 0.2
        self.sgd_minibatch_size = 500
        self.num_sgd_iter = 20
        self.shuffle_sequences = True
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.001
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = 0.0
        self.grad_clip = None
        self.kl_target = 0.03
        self.seq_len = 10

        # Override some of PG/AlgorithmConfig's default values with PPO-specific values.
        self.num_rollout_workers = 20
        self.train_batch_size = 4000
        self.lr = 5e-5
        self.model["vf_share_layers"] = False
        self._disable_preprocessor_api = False
        # __sphinx_doc_end__
        # fmt: on
        self.replay_buffer_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            "prioritized_replay": DEPRECATED_VALUE,
            # Size of the replay buffer. Note that if async_updates is set,
            # then each worker will have a replay buffer of this size.
            "capacity": 1000000,
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # The number of continuous environment steps to replay at once. This may
            # be set to greater than 1 to support recurrent models.
            "storage_unit": StorageUnit.SEQUENCES,
            "replay_sequence_length": 5,
            "replay_burn_in": 0,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
        }

        

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        seq_len: Optional[int] = NotProvided,
        # Deprecated.
        vf_share_layers=DEPRECATED_VALUE,
        **kwargs,
    ) -> "MBPPOConfig":
        """Sets the training related configuration.

        Args:
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The GAE (lambda) parameter.
            kl_coeff: Initial coefficient for KL divergence.
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
                This defines the minibatch size within each epoch.
            num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of
                epochs to execute per train batch).
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: Coefficient of the entropy regularizer.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            clip_param: PPO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            kl_target: Target value for KL divergence.

        Returns:
            This updated AlgorithmConfig object.
        """
        if vf_share_layers != DEPRECATED_VALUE:
            deprecation_warning(
                old="ppo.DEFAULT_CONFIG['vf_share_layers']",
                new="PPOConfig().training(model={'vf_share_layers': ...})",
                error=True,
            )

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            if isinstance(entropy_coeff, int):
                entropy_coeff = float(entropy_coeff)
            if entropy_coeff < 0.0:
                raise ValueError("`entropy_coeff` must be >= 0.0")
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if kl_target is not NotProvided:
            self.kl_target = kl_target
        if seq_len is not NotProvided:
            self.seq_len = seq_len

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        # Note: Only check this if `train_batch_size` > 0 (DDPPO sets this
        # to -1 to auto-calculate the actual batch size later).
        if self.sgd_minibatch_size > self.train_batch_size:
            raise ValueError(
                f"`sgd_minibatch_size` ({self.sgd_minibatch_size}) must be <= "
                f"`train_batch_size` ({self.train_batch_size}). In PPO, the train batch"
                f" is be split into {self.sgd_minibatch_size} chunks, each of which is "
                f"iterated over (used for updating the policy) {self.num_sgd_iter} "
                "times."
            )

        # Episodes may only be truncated (and passed into PPO's
        # `postprocessing_fn`), iff generalized advantage estimation is used
        # (value function estimate at end of truncated episode to estimate
        # remaining value).
        if (
            not self.in_evaluation
            and self.batch_mode == "truncate_episodes"
            and not self.use_gae
        ):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function (to estimate the return at the end of the truncated"
                " trajectory). Consider setting "
                "batch_mode=complete_episodes."
            )


class UpdateKL:
    """Callback to update the KL based on optimization info.

    This is used inside the execution_plan function. The Policy must define
    a `update_kl` method for this to work. This is achieved for PPO via a
    Policy mixin class (which adds the `update_kl` method),
    defined in ppo_[tf|torch]_policy.py.
    """

    def __init__(self, workers):
        self.workers = workers

    def __call__(self, fetches):
        def update(pi, pi_id):
            assert LEARNER_STATS_KEY not in fetches, (
                "{} should be nested under policy id key".format(LEARNER_STATS_KEY),
                fetches,
            )
            if pi_id in fetches:
                kl = fetches[pi_id][LEARNER_STATS_KEY].get("kl")
                assert kl is not None, (fetches, pi_id)
                # Make the actual `Policy.update_kl()` call.
                pi.update_kl(kl)
            else:
                logger.warning("No data for {}, not updating kl".format(pi_id))

        # Update KL on all trainable policies within the local (trainer)
        # Worker.
        self.workers.local_worker().foreach_policy_to_train(update)


class MBPPO(Algorithm):

    def __init__(self, config, logger_creator) -> None:
        super().__init__(config, logger_creator)
        self.memeory = {}
        self.memeory['obs_action_hist'] = None
        self.reward_to_index = np.load('reward_to_index.npy', allow_pickle=True).item()
        self.index_to_reward = np.load('index_to_reward.npy', allow_pickle=True).item()
        self.wm_train_interval = 0
        self.seq_len = config['seq_len']
        self.known_states = None
        self.local_replay_buffer = None

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return MBPPOConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            assert 'tf' in config["framework"], ("Only tf is supported")
        elif config["framework"] == "tf":
            from MBPPO_tf_policy import MBPPOTF1Policy
            return MBPPOTF1Policy
        else:
            from MBPPO_tf_policy import MBPPOTF2Policy
            return MBPPOTF2Policy
        
    def encode_batch(self, batch, dream):
        if dream:
            np.save('batch_encodings/dream_obs_' + str(self._counters[NUM_AGENT_STEPS_SAMPLED]) + '.npy',
                np.mean(batch['obs'], axis=0))
        else:
            np.save('batch_encodings/obs_' + str(self._counters[NUM_AGENT_STEPS_SAMPLED]) + '.npy',
                    np.mean(batch['obs'], axis=0))

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )
            
        self.encode_batch(train_batch, False)
        
        self.process_experience_lstm(train_batch)
        #self.local_replay_buffer.add(train_batch)

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        train_results = self.learning_from_samples(train_batch)

        dreams_start_at = 100000
   
        if self._counters[NUM_AGENT_STEPS_SAMPLED] > dreams_start_at:
            if self.wm_train_interval == 0:
                #experience_data = self.local_replay_buffer.sample(self._counters[NUM_AGENT_STEPS_SAMPLED])
                #for pid in self.workers.local_worker().get_policies_to_train({'default_policy'}):
                for pid in {'default_policy'}: #Could extend to multiagent here
                    self.workers.local_worker().get_policy(pid).reward_model.fit(self.memeory['obs_action_hist'], self.memeory['next_obs'], self.memeory['rewards'])
                    self.workers.local_worker().get_policy(pid).state_tranistion_model.fit(self.memeory['node_ids'], self.memeory['node_vectors'], self.memeory['node_predictions'], self.memeory['node_predictions2'], self.memeory['next_nodes'])

                    #Sync
                    reward_weights = self.workers.local_worker().get_policy(pid).reward_model.base_model.get_weights()
                    def set_reward_weights(policy, pid):
                        policy.reward_model.base_model.set_weights(reward_weights)
                    self.workers.foreach_policy(set_reward_weights)

                    state_weights = self.workers.local_worker().get_policy(pid).state_tranistion_model.get_weights()
                    known_states = list(self.known_states)
                    def set_state_weights(policy, pid):
                        policy.state_tranistion_model.set_weights(state_weights, known_states)
                    self.workers.foreach_policy(set_state_weights)

                self.wm_train_interval = 0
            else: 
                self.wm_train_interval -= 1

        #Dream
        if self._counters[NUM_AGENT_STEPS_SAMPLED] > dreams_start_at:
            s = []; 
            for d in range(1):
                for i in range(1):
                    samples = self.workers.foreach_policy(dream)
                    s.append(SampleBatch.concat_samples(samples))
                samples = SampleBatch.concat_samples(s)
                print('Dream Reward Mean: ', np.sum(samples['rewards'])/20)
                self.encode_batch(samples, True)
                self.learning_from_samples(samples.as_multi_agent())
        
        return train_results

    def learning_from_samples(self, samples):
          # Standardize advantages
        train_batch = standardize_fields(samples, ["advantages"])
        # Train
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)
        return train_results
    
    def process_experience_lstm(self, samples):    
        #TODO vectorises this
        STATE_LEN = 91
        #obs_seq = np.zeros((samples['dones'].shape[0]-samples['dones'].sum(), self.seq_len, STATE_LEN))
        node_vectors = np.zeros((samples['dones'].shape[0]*13-samples['dones'].sum()*13, self.seq_len, STATE_LEN+3), dtype=np.int8)
        node_ids = np.zeros((samples['dones'].shape[0]*13-samples['dones'].sum()*13, 13), dtype=np.int8)
        next_obs = np.zeros((samples['dones'].shape[0]-samples['dones'].sum(), STATE_LEN), dtype=np.int8)
        #actions_hist = np.zeros(((samples['dones'].shape[0]-samples['dones'].sum()), self.seq_len, 41))
        obs_action_hist = np.zeros(((samples['dones'].shape[0]-samples['dones'].sum()), self.seq_len, STATE_LEN+41), dtype=np.int8)
        rewards = np.zeros((samples['dones'].shape[0]-samples['dones'].sum(), len(self.reward_to_index.keys())), dtype=np.int8)
        next_nodes = np.zeros((samples['dones'].shape[0]*13-samples['dones'].sum()*13, 7), dtype=np.int8)
        node_predictions = np.zeros((samples['dones'].shape[0]*13-samples['dones'].sum()*13, STATE_LEN), dtype=np.int8)
        node_predictions2 = np.zeros((samples['dones'].shape[0]*13-samples['dones'].sum()*13, STATE_LEN), dtype=np.int8)
        #next_obs = np.zeros((samples['dones'].shape[0]-samples['dones'].sum(), STATE_LEN))

        s_index = 0; index = 0; ts = 0
        for i in range(samples['dones'].shape[0]-1):
            steps = ts if ts < self.seq_len else self.seq_len-1     
            if samples['dones'][i]:
                ts = 0
                continue
            next_obs[s_index,:] = samples['obs'][i+1,:]
            obs_action_hist[s_index,:steps+1,:STATE_LEN] = samples['obs'][i-steps:i+1,:]
            obs_action_hist[s_index,:steps+1,STATE_LEN:] = np.eye(41)[samples['actions'][i-steps:i+1]]
            reward = self.take_closest(list(self.reward_to_index.keys()), samples['rewards'][i])
            rewards[s_index,self.reward_to_index[reward]] = 1        
            s_index += 1
            for n in range(13):     
                node_ids[index,n] = 1
                node_vectors[index,:steps+1,:STATE_LEN] = samples['obs'][i-steps:i+1,:]
                node_vectors[index,:steps+1,STATE_LEN:] = np.array([self.node_action(samples['actions'][i], n) for i in range(i-steps,i+1)])

                # node_vectors[index,:steps+1,:7] = samples['obs'][i-steps:i+1,int(n*7):int(n*7)+7]
                # node_vectors[index,:steps+1,7:10] = np.array([self.node_action(samples['actions'][i], n) for i in range(i-steps,i+1)])
                # #exploit = 0
                # node_vectors[index,:steps+1,10:23] = samples['obs'][i-steps:i+1,np.arange(0,91,step=7)]
                # # #scan = 1
                # # #privilege = 3
                # node_vectors[index,:steps+1,23:36] = samples['obs'][i-steps:i+1,np.arange(3,91,step=7)]
                # # #user = 4
                # node_vectors[index,:steps+1,36:49] = samples['obs'][i-steps:i+1,np.arange(4,91,step=7)]
                # # #unknown = 5 
                # node_vectors[index,:steps+1,49:] = samples['obs'][i-steps:i+1,np.arange(5,91,step=7)]
                # # #no = 6
                next_nodes[index,:] = samples['obs'][i+1,int(n*7):int(n*7)+7]
                index += 1
            ts += 1

        #Todo
        for i in range(next_nodes.shape[0]):
            index = (i // 13) * 13
            range_ = i % 13
            if range_ > 0:
                node_predictions[i,:int(range_*7)] = next_nodes[index:index+range_,:].reshape(-1)
                node_predictions2[i,:int(range_*7)+3] = np.concatenate([next_nodes[index:index+range_,:].reshape(-1), next_nodes[index+range_,:3]])
            else:
                node_predictions2[i,:3] = next_nodes[index+range_,:3]

        if type(self.memeory['obs_action_hist']) == type(None):
            self.memeory['obs_action_hist'] = obs_action_hist
            self.memeory['next_obs'] = next_obs
            self.memeory['rewards'] = rewards
            self.memeory['node_vectors'] = node_vectors
            self.memeory['next_nodes'] = next_nodes
            self.memeory['node_ids'] = node_ids
            self.memeory['node_predictions'] = node_predictions
            self.memeory['node_predictions2'] = node_predictions2
        else:
            self.memeory['obs_action_hist'] = np.concatenate((self.memeory['obs_action_hist'], obs_action_hist))
            self.memeory['next_obs'] = np.concatenate((self.memeory['next_obs'], next_obs))
            self.memeory['rewards'] = np.concatenate((self.memeory['rewards'], rewards))
            self.memeory['node_vectors'] = np.concatenate((self.memeory['node_vectors'], node_vectors))
            self.memeory['next_nodes'] = np.concatenate((self.memeory['next_nodes'], next_nodes))
            self.memeory['node_ids'] = np.concatenate((self.memeory['node_ids'], node_ids))
            self.memeory['node_predictions'] = np.concatenate((self.memeory['node_predictions'], node_predictions))
            self.memeory['node_predictions2'] = np.concatenate((self.memeory['node_predictions2'], node_predictions2))

        self.known_states = np.unique(self.memeory['next_obs'], axis=0)
    
    from bisect import bisect_left

    def take_closest(self, myList, myNumber):

        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before
     
    def node_action(self, action, node):
        vec = np.zeros(3)
        if action < 2: 
            return vec
        action -= 2
        if action % node == 0:
            #Analyse #Remove #Resotre
            vec[int(action) // 13] = 1
        return vec


def dream(policy, pid):
    return policy.fetch_dream_lstm()

# Deprecated: Use ray.rllib.algorithms.ppo.PPOConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(MBPPOConfig().to_dict())

    @Deprecated(
        old="ray.rllib.agents.ppo.ppo::DEFAULT_CONFIG",
        new="ray.rllib.algorithms.ppo.ppo::PPOConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()