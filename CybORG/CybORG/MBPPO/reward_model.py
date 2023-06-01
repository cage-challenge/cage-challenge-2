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
        self.max_train_epochs = 60
       # super().__init__()

        input_ = Input(shape=(self.SEQ_LEN ,132), name='state_action')
        new_state_in = Input(shape=(self.STATE_LEN,),name='state_in')
        x = Bidirectional(LSTM(64))(input_)
        x = concatenate([x, new_state_in], name='concate')
        x = Dense(128, activation='relu', name='hidden')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', name='hidden2')(x)
        x = Dropout(0.2)(x)
        out = Dense(1)(x)

        def scheduler(epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * tf.math.exp(-0.03)

        self.base_model = Model([input_, new_state_in], out)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001, restore_best_weights=True)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError())#, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def forward(self, x, ns):
        return -self.base_model([x, ns]).numpy().reshape(-1)
    
    def load(self, path):
        self.base_model.load_weights(path)
        
    def fit(self, obs, ns, rewards): 
        # Process Samples
        p = np.random.permutation(obs.shape[0])
        K.set_value(self.base_model.optimizer.learning_rate, 0.0002)
        print(obs.shape)
        print(ns.shape)
        print(rewards.shape)
        try:
            with tf.device("/device:CPU:35"):
                history = self.base_model.fit([obs[p,:,:], ns[p,:]], rewards[p], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                            verbose=0, callbacks=[self.callback, self.lr_callback], batch_size=200, shuffle=True)
                #history = self.base_model.fit(train_dataset, validation_data=val_dataset, epochs=self.max_train_epochs, 
                #                             verbose=0, callbacks=[self.callback])
            K.clear_session()
            print('reward val loss: ', history.history['val_loss'])
            #print('reward accuracy ', history.history['val_categorical_accuracy'])
            self.global_itr += 1
                # Returns Metric Dictionary
        except Exception as e:
            print('reward train fail')
            print(e)
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