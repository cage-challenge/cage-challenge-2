"""
TensorFlow policy class used for PPO.
"""

import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import explained_variance, warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType, TFPolicyV2Type
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from state_tranistion_model import CAGEStateTranistionModel
from node_tranistion_model import CAGENodeTranistionModel
from lstm_node_tranistion_model import CAGENodeTranistionModelLSTM
from lstm_node_tranistion_model_feedback2 import CAGENodeTranistionModelLSTMFeedback2
from reward_model import CAGERewardModel
from reward_model_lstm import CAGERewardModelLSTM

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) 

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


def validate_config(config: AlgorithmConfigDict) -> None:
    """Executed before Policy is "initialized" (at beginning of constructor).
    Args:
        config: The Policy's config.
    """
    # If vf_share_layers is True, inform about the need to tune vf_loss_coeff.
    if config.get("model", {}).get("vf_share_layers") is True:
        logger.info(
            "`vf_share_layers=True` in your model. "
            "Therefore, remember to tune the value of `vf_loss_coeff`!"
        )


# We need this builder function because we want to share the same
# custom logics between TF1 dynamic and TF2 eager policies.
def get_mbppo_tf_policy(name: str, base: TFPolicyV2Type) -> TFPolicyV2Type:
    """Construct a PPOTFPolicy inheriting either dynamic or eager base policies.

    Args:
        base: Base class for this policy. DynamicTFPolicyV2 or EagerTFPolicyV2.

    Returns:
        A TF Policy to be used with PPO.
    """

    class MBPPOTFPolicy(
        EntropyCoeffSchedule,
        LearningRateSchedule,
        KLCoeffMixin,
        ValueNetworkMixin,
        base,
    ):
        def __init__(
            self,
            observation_space,
            action_space,
            config,
            existing_model=None,
            existing_inputs=None,
        ):
            # First thing first, enable eager execution if necessary.
            base.enable_eager_execution_if_necessary()

            config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
            # TODO: Move into Policy API, if needed at all here. Why not move this into
            #  `PPOConfig`?.
            validate_config(config)

            # Initialize base class.
            base.__init__(
                self,
                observation_space,
                action_space,
                config,
                existing_inputs=existing_inputs,
                existing_model=existing_model,
            )

            # Initialize MixIns.
            ValueNetworkMixin.__init__(self, config)
            KLCoeffMixin.__init__(self, config)
            EntropyCoeffSchedule.__init__(
                self, config["entropy_coeff"], config["entropy_coeff_schedule"]
            )
            LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

            # Note: this is a bit ugly, but loss and optimizer initialization must
            # happen after all the MixIns are initialized.
            self.maybe_initialize_optimizer_and_loss()
            self.observation_length = observation_space.shape[0]
            self.action_length = action_space.n
            self.inital_state = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])
            
            self.NUM_NODES = 13
            self.NODE_CLASSES = [3, 4]
            self.STATE_LEN = 91
            self.ACTION_LEN = 41
            self.SEQ_LEN = config["seq_len"]
            self.make_world_model()
            
        def make_world_model(self):
            self.state_tranistion_model = CAGENodeTranistionModelLSTMFeedback2(self.SEQ_LEN)#CAGENodeTranistionModel()#CAGENodeTranistionModel()  CAGEStateTranistionModel
            self.reward_model = CAGERewardModel(self.SEQ_LEN)

        
        #TODO this can probably be parallelised to run simultaneous dreams on one thread? 
        #also fix all the array shapes on calls
        def fetch_dream_lstm(self):
            obs = np.zeros((100, self.STATE_LEN))
            actions = np.zeros(100, dtype=np.float64)
            action_dist_inputs = np.zeros((100, self.ACTION_LEN), dtype=np.float64)
            values = np.zeros(100)
            action_logps = np.zeros(100, dtype=np.float64)
            rewards = np.zeros(100)
            dones = np.zeros(100)
            dones[-1] = 1
            states = np.zeros((self.SEQ_LEN, self.STATE_LEN))
            states[0,:] = self.inital_state
            actions_seq = np.zeros(self.SEQ_LEN)
            actions_onehot = np.zeros((self.SEQ_LEN, 41))
            state = self.inital_state
            for i in range(100):
                obs[i,:] = states[0,:]
                action, _, info = self.compute_actions(np.array([state]), explore=True)
                if i < self.SEQ_LEN:
                    actions_seq[i] = action
                    actions_onehot[i,action] = 1
                else:
                    actions_seq = np.roll(actions_seq,-1,axis=0)
                    actions_seq[-1] = action         
                    actions_onehot = np.roll(actions_onehot,-1,axis=0)
                    actions_onehot[-1,action] = 1

                actions[i] = np.float64(action[0])
                action_logps[i] = np.float64(info['action_logp'][0])
                action_dist_inputs[i,:] = info['action_dist_inputs'][0]
                values[i] = info['vf_preds'][0]
                state = self.state_tranistion_model.forward(states, actions_seq)

                state_action = np.concatenate([states, actions_onehot], axis=-1)

                rewards[i] = self.reward_model.forward(np.array([state_action]), np.array([state]))
                if i < self.SEQ_LEN -1:
                    states[i+1] = state
                else:
                    states = np.roll(states,-1,axis=0)
                    states[-1] = state
                    
            batch = SampleBatch({'obs': obs,
                                'actions': actions,
                                'rewards': rewards,
                                'vf_preds': values,
                                'action_dist_inputs': action_dist_inputs,
                                'action_logp': action_logps,
                                'dones': dones})

            return compute_advantages(batch, values[-1], self.config['gamma'], self.config['lambda'], 
                                        self.config["use_gae"], self.config['use_critic'])
        

        def fetch_dream(self):
            obs = np.zeros((100, self.STATE_LEN))
            actions = np.zeros(100, dtype=np.float64)
            action_dist_inputs = np.zeros((100, self.ACTION_LEN), dtype=np.float64)
            values = np.zeros(100)
            action_logps = np.zeros(100, dtype=np.float64)
            rewards = np.zeros(100)
            dones = np.zeros(100)
            dones[-1] = 1
            state = self.inital_state
            for i in range(100):
                obs[i,:] = state
                action, _, info = self.compute_actions(np.array([state]), explore=True)
                actions[i] = np.float64(action[0])
                action_logps[i] = np.float64(info['action_logp'][0])
                action_dist_inputs[i,:] = info['action_dist_inputs'][0]
                values[i] = info['vf_preds'][0]
                #action_one_hot = np.zeros(self.ACTION_LEN)
                #action_one_hot[action[0]] = 1
                if i == 0:
                    prev_action = 0
                else:
                    prev_action = actions[i-1]
                state = self.state_tranistion_model.forward(np.array([np.concatenate([state, [i/100], [action[0]]])]), prev_action)
                #state = self.state_tranistion_model.forward(np.array([np.concatenate([state, [i/100], action_one_hot])]))
                #rewards[i] = self.reward_model.forward(np.array([np.concatenate([obs[i,:], [i/100], state])]))
                a = np.zeros(41)
                a[int(action[0])] = 1
                rewards[i] = self.reward_model.forward(np.array([np.concatenate([obs[i,:], a])]))
                #rewards[i] = self.reward_model.forward(np.array([state]))

            batch = SampleBatch({'obs': obs,
                                 'actions': actions,
                                 'rewards': rewards,
                                 'vf_preds': values,
                                 'action_dist_inputs': action_dist_inputs,
                                 'action_logp': action_logps,
                                 'dones': dones})

            return compute_advantages(batch, values[-1], self.config['gamma'], self.config['lambda'], 
                                      self.config["use_gae"], self.config['use_critic'])
        
        def fetch_dreams(self):
            num_episodes = 5
            obs = np.zeros((num_episodes, 100, self.STATE_LEN))
            next_obs = np.zeros((num_episodes, 100, self.STATE_LEN))
            actions = np.zeros(num_episodes, 100, dtype=np.float64)
            action_dist_inputs = np.zeros((num_episodes, 100, self.ACTION_LEN), dtype=np.float64)
            values = np.zeros((num_episodes, 100))
            action_logps = np.zeros((num_episodes, 100), dtype=np.float64)
            rewards = np.zeros((num_episodes, 100))
            dones = np.zeros((num_episodes, 100))
            dones[:,-1] = 1
            state = np.repeat(self.inital_state, num_episodes).reshape(-1,num_episodes).transpose()
            for i in range(100):
                obs[i,:] = state
                action, _, info = self.compute_actions(state, explore=False)
                actions[:,i] = np.float64(action[0])
                action_logps[:,i] = np.float64(info['action_logp'][0])
                action_dist_inputs[:,i,:] = info['action_dist_inputs'][0]
                values[:,i] = info['vf_preds'][0]
                action_one_hot = np.zeros(num_episodes, self.ACTION_LEN)
                action_one_hot[action[0]] = 1
                state = self.state_tranistion_model.forward(np.concatenate([state, action_one_hot]))
                next_obs[i,:] = state
                #rewards[:,i] = self.reward_model.forward(np.concatenate([obs[i,:], state]))
                rewards[:,i] = self.reward_model.forward(obs[i,:])

            batch = SampleBatch({'obs': obs.reshape(-1, self.STATE_LEN),
                                 'actions': actions.reshape(-1),
                                 'rewards': rewards.reshape(-1),
                                 'vf_preds': values.reshape(-1),
                                 'action_dist_inputs': action_dist_inputs.reshape(-1, self.ACTION_LEN),
                                 'action_logp': action_logps.reshape(-1),
                                 'dones': dones.reshape(-1)})

            return compute_advantages(batch, values[-1], self.config['gamma'], self.config['lambda'], 
                                      self.config["use_gae"], self.config['use_critic'])

        @override(base)
        def loss(
            self,
            model: Union[ModelV2, "tf.keras.Model"],
            dist_class: Type[TFActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
            if isinstance(model, tf.keras.Model):
                logits, state, extra_outs = model(train_batch)
                value_fn_out = extra_outs[SampleBatch.VF_PREDS]
            else:
                logits, state = model(train_batch)
                value_fn_out = model.value_function()

            curr_action_dist = dist_class(logits, model)

            # RNN case: Mask away 0-padded chunks at end of time axis.
            if state:
                # Derive max_seq_len from the data itself, not from the seq_lens
                # tensor. This is in case e.g. seq_lens=[2, 3], but the data is still
                # 0-padded up to T=5 (as it's the case for attention nets).
                B = tf.shape(train_batch[SampleBatch.SEQ_LENS])[0]
                max_seq_len = tf.shape(logits)[0] // B

                mask = tf.sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
                mask = tf.reshape(mask, [-1])

                def reduce_mean_valid(t):
                    return tf.reduce_mean(tf.boolean_mask(t, mask))

            # non-RNN case: No masking.
            else:
                mask = None
                reduce_mean_valid = tf.reduce_mean

            prev_action_dist = dist_class(
                tf.cast(train_batch[SampleBatch.ACTION_DIST_INPUTS], tf.float32), model
            )

            logp_ratio = tf.exp(
                curr_action_dist.logp(tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32))
                - tf.cast(train_batch[SampleBatch.ACTION_LOGP], tf.float32)
            )

            # Only calculate kl loss if necessary (kl-coeff > 0.0).
            if self.config["kl_coeff"] > 0.0:
                action_kl = prev_action_dist.kl(curr_action_dist)
                mean_kl_loss = reduce_mean_valid(action_kl)
                warn_if_infinite_kl_divergence(self, mean_kl_loss)
            else:
                mean_kl_loss = tf.constant(0.0)

            curr_entropy = curr_action_dist.entropy()
            mean_entropy = reduce_mean_valid(curr_entropy)

            surrogate_loss = tf.minimum(
                train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]
                * tf.clip_by_value(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            # Compute a value function loss.
            if self.config["use_critic"]:
                vf_loss = tf.math.square(
                    value_fn_out - train_batch[Postprocessing.VALUE_TARGETS]
                )
                vf_loss_clipped = tf.clip_by_value(
                    vf_loss,
                    0,
                    self.config["vf_clip_param"],
                )
                mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            # Ignore the value function.
            else:
                vf_loss_clipped = mean_vf_loss = tf.constant(0.0)

            total_loss = reduce_mean_valid(
                -surrogate_loss
                + self.config["vf_loss_coeff"] * vf_loss_clipped
                - self.entropy_coeff * curr_entropy
            )
            # Add mean_kl_loss (already processed through `reduce_mean_valid`),
            # if necessary.
            if self.config["kl_coeff"] > 0.0:
                total_loss += self.kl_coeff * mean_kl_loss

            # Store stats in policy for stats_fn.
            self._total_loss = total_loss
            self._mean_policy_loss = reduce_mean_valid(-surrogate_loss)
            self._mean_vf_loss = mean_vf_loss
            self._mean_entropy = mean_entropy
            # Backward compatibility: Deprecate self._mean_kl.
            self._mean_kl_loss = self._mean_kl = mean_kl_loss
            self._value_fn_out = value_fn_out

            return total_loss

        @override(base)
        def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
            return {
                "cur_kl_coeff": tf.cast(self.kl_coeff, tf.float64),
                "cur_lr": tf.cast(self.cur_lr, tf.float64),
                "total_loss": self._total_loss,
                "policy_loss": self._mean_policy_loss,
                "vf_loss": self._mean_vf_loss,
                "vf_explained_var": explained_variance(
                    train_batch[Postprocessing.VALUE_TARGETS], self._value_fn_out
                ),
                "kl": self._mean_kl_loss,
                "entropy": self._mean_entropy,
                "entropy_coeff": tf.cast(self.entropy_coeff, tf.float64),
            }

        @override(base)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            sample_batch = super().postprocess_trajectory(sample_batch)
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )


    MBPPOTFPolicy.__name__ = name
    MBPPOTFPolicy.__qualname__ = name

    return MBPPOTFPolicy


MBPPOTF1Policy = get_mbppo_tf_policy("PPOTF1Policy", DynamicTFPolicyV2)
MBPPOTF2Policy = get_mbppo_tf_policy("PPOTF2Policy", EagerTFPolicyV2)