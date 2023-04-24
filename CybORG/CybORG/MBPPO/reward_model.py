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
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional, concatenate
from tensorflow.keras import backend as K

class CAGERewardModel(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(self, seq_len):
        
        self.NUM_NODES = 13
        self.NODE_CLASSES = [3, 4]
        self.STATE_LEN = 91
        self.ACTION_LEN = 41
        self.SEQ_LEN = seq_len
        self.global_itr = 0
        self.valid_split = 0.2
        self.max_train_epochs = 50
        self.reward_to_index = np.load('/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/MBPPO/reward_to_index.npy', allow_pickle=True).item()
        self.index_to_reward = np.load('/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/MBPPO/index_to_reward.npy', allow_pickle=True).item()
        self.number_rewards = int(len(self.reward_to_index.keys()))
        print(self.number_rewards )
       # super().__init__()

        input_ = Input(shape=(self.SEQ_LEN, self.STATE_LEN+self.ACTION_LEN, ), name='state_action')
        new_state_in = Input(shape=(self.STATE_LEN,),name='state_in')
        x = Bidirectional(LSTM(64))(input_)
        x = Flatten()(x)
        x = concatenate([x, new_state_in], name='concate')
        x = Dense(128, activation='relu', name='hidden')(x)
        x = Dropout(0.2)(x)
        out = Dense(self.number_rewards, activation='softmax')(x)

        def scheduler(epoch, lr):
            if epoch < 2:
                return lr
            else:
                return lr * tf.math.exp(-0.05)

        self.base_model = Model([input_, new_state_in], out)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0005, restore_best_weights=True)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def forward(self, x, ns):
        probs = self.base_model([x, ns]).numpy()[0]
        index = np.random.choice(np.arange(self.number_rewards), p=probs)
        probs[probs==0] = 1e-8
        self.entropy = - np.sum(np.log(probs) * probs) / probs.shape[0]
        return self.index_to_reward[index]
    
    def load(self, path):
        self.base_model.load_weights(path)

    def get_entropy(self):
        return self.entropy
        
    def fit(self, obs, ns, rewards): 
        # Process Samples
        print(obs.shape)
        p = np.random.permutation(obs.shape[0])
        try:
            with tf.device("/device:GPU:1"):
                history = self.base_model.fit([obs[p,:,:], ns[p,:]], rewards[p], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                                verbose=0, callbacks=[self.callback, self.lr_callback], batch_size=256, shuffle=True, workers=2)
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