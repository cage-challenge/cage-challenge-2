import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input
from bisect import bisect_left
from tqdm import tqdm

#from ProcessTrueStateActionData import read_df_in_chunks

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

train_test_split = 0.75
data_path = '/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/Notebooks/logs/PPO/no_decoy_200000'
state = np.load(data_path + '/data/state.npy')
rewards = np.load(data_path + '/data/rewards.npy')
next_state = np.load(data_path + '/data/next_state.npy')
actions = np.load(data_path + '/data/actions.npy')
print(actions.shape)

data = np.concatenate([state, next_state], axis=1)

reward_to_index = np.load('reward_to_index.npy', allow_pickle=True).item()
index_to_reward = np.load('index_to_reward.npy', allow_pickle=True).item()
number_rewards = int(len(reward_to_index.keys()))

for i in range(rewards.shape[0]):
    rewards[i] = take_closest(list(reward_to_index.keys()), rewards[i])
reward_classes = np.vectorize(reward_to_index.get)(rewards)
reward_onehot = np.eye(int(len(reward_to_index.keys())))[np.array(reward_classes, dtype=np.int8)]
    

STATE_LEN = 91
ACTION_LEN = 3
    
max_train_epochs = 50

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

input_ = Input(shape=(data.shape[1],))
x = Dense(128, activation='relu')(input_)
x = Dense(128, activation='relu')(x)
#x = Dense(256, activation='relu')(x)
out = Dense(number_rewards, activation='softmax')(x)
base_model = Model(input_, out)

base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

p = np.random.permutation(reward_onehot.shape[0])
with tf.device("/device:GPU:0"):
    history = base_model.fit(data[p,:], reward_onehot[p], epochs=max_train_epochs, validation_split=0.5, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True)
    
base_model.save_weights('reward_model')
