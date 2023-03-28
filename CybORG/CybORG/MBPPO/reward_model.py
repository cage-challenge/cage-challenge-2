import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from keras.models import Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Input
import tensorflow as tf

class CAGERewardModel(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(
        self,
      #  input_size,
      #  output_size,
      #  hidden_layers=(512, 512),
      #  hidden_nonlinearity=None,
      #  output_nonlinearity=None,
      #  weight_normalization=False,
      #  use_bias=True,
    ):
        
        self.NUM_NODES = 13
        self.NODE_CLASSES = [3, 4]
        self.STATE_LEN = 91
        self.ACTION_LEN = 41

        self.global_itr = 0
        self.valid_split = 0.1
        self.max_train_epochs = 50
        self.reward_to_index = np.load('reward_to_index.npy', allow_pickle=True).item()
        self.index_to_reward = np.load('index_to_reward.npy', allow_pickle=True).item()
        self.number_rewards = int(len(self.reward_to_index.keys()))
       # super().__init__()

        input_ = Input(shape=(self.STATE_LEN,))

        x = Dense(256, activation='tanh')(input_)
        x = Dense(256, activation='tanh')(x)
        #x = Dense(256, activation='relu')(x)
        out = Dense(self.number_rewards, activation='softmax')(x)

        self.base_model = Model(input_, out)
       
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.005)

    def forward(self, x):
        probs = self.base_model(x).numpy()[0]
        index = np.random.choice(np.arange(self.number_rewards), p=probs)
        ##index = np.random.categorical(probs, 1).numpy()
        return self.index_to_reward[index]
        
    
    def fit(self, samples): 
        # Process Samples
        samples = self.process_samples(samples)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        with tf.device("/device:GPU:0"):
            history = self.base_model.fit(samples['obs_concat'], samples['rewards'], epochs=self.max_train_epochs, validation_split=self.valid_split, verbose=0, callbacks=[self.callback], batch_size=256)
        print('reward val loss: ', history.history['val_loss'])
        print('reward accuracy ', history.history['val_categorical_accuracy'])
        self.global_itr += 1
        # Returns Metric Dictionary
        return self.metrics
        
    def process_samples(self, samples: SampleBatchType):
        processed = {}
        for i, r in enumerate(samples['rewards']):
            samples['rewards'][i] = take_closest(list(self.reward_to_index.keys()), r)
        reward_classes = np.vectorize(self.reward_to_index.get)(samples['rewards'])
        reward_onehot = np.eye(int(len(self.reward_to_index.keys())))[np.array(reward_classes, dtype=np.int8)]
        processed['rewards'] = reward_onehot
        processed['obs_concat'] = np.concatenate([samples['obs'], samples['next_obs']], axis=1)
        processed['obs_concat'] = samples['next_obs']
        return SampleBatch(processed)

from bisect import bisect_left

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
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