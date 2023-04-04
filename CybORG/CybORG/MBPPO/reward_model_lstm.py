import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional
from tensorflow.keras import backend as K

class CAGERewardModelLSTM(TFModelV2):
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
        self.valid_split = 0.2
        self.max_train_epochs = 50
        self.reward_to_index = np.load('reward_to_index.npy', allow_pickle=True).item()
        self.index_to_reward = np.load('index_to_reward.npy', allow_pickle=True).item()
        self.number_rewards = int(len(self.reward_to_index.keys()))
       # super().__init__()

        input_ = Input(shape=(10,self.STATE_LEN,))
        x = Bidirectional(LSTM(128))(input_)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', name='hidden')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', name='hidden2')(x)
        out = Dense(self.number_rewards, activation='softmax')(x)

        def scheduler(epoch, lr):
            if epoch < 2:
                return lr
            else:
                return lr * tf.math.exp(-0.05)


        self.base_model = Model(input_, out)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01, restore_best_weights=True)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def forward(self, x):
        probs = self.base_model(x).numpy()[0]
        index = np.random.choice(np.arange(self.number_rewards), p=probs)
        ##index = np.random.categorical(probs, 1).numpy()
        return self.index_to_reward[index]
        
    
    def fit(self, obs, rewards): 
        # Process Samples
        print(obs.shape)
        #samples = self.process_samples(obs, next_obs, rewards)
        p = np.random.permutation(obs.shape[0])
        #val_size = int(0.2 * obs.shape[0])
        #train_size = int(0.8 * obs.shape[0])
        #val_dataset = dataset.skip(train_size)
        #train_dataset = dataset.take(train_size)
        #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128, drop_remainder=True)
        #val_dataset  = val_dataset.shuffle(buffer_size=1024).batch(128, drop_remainder=True)
        try:
            with tf.device("/device:GPU:0"):
                history = self.base_model.fit(obs[p,:,:], rewards[p], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                                verbose=0, callbacks=[self.callback, self.lr_callback], batch_size=256, shuffle=True, workers=4)
                #history = self.base_model.fit(train_dataset, validation_data=val_dataset, epochs=self.max_train_epochs, 
                #                             verbose=0, callbacks=[self.callback])
            K.clear_session()
            print('reward val loss: ', history.history['val_loss'])
            print('reward accuracy ', history.history['val_categorical_accuracy'])
            self.global_itr += 1
                # Returns Metric Dictionary
        except:
            print('reward train fail')
        return self.metrics
        
    def process_samples(self, obs, next_obs, rewards):
        processed = {}
        for i, r in enumerate(rewards):
            rewards[i] = take_closest(list(self.reward_to_index.keys()), r)
        reward_classes = np.vectorize(self.reward_to_index.get)(rewards)
        reward_onehot = np.eye(int(len(self.reward_to_index.keys())))[np.array(reward_classes, dtype=np.int8)]
        processed['rewards'] = reward_onehot
        #processed['obs_concat'] = np.concatenate([obs, next_obs], axis=1)
        processed['obs_concat'] = obs
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