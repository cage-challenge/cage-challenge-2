import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional
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
data_path = '/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/Notebooks/logs/PPO/no_decoy_200000/data_seqence_5'
state = np.load(data_path + '/states.npy')
rewards = np.load(data_path + '/rewards.npy')
next_state = np.load(data_path + '/next_states.npy')

labels, encoding = np.unique(rewards, return_inverse=True)
index_to_reward = {}; reward_to_index = {}
for i in range(labels.shape[0]):
    index_to_reward[i] = labels[i]
    reward_to_index[labels[i]] = i
#np.save('index_to_reward.npy', index_to_reward) 
#np.save('reward_to_index.npy', reward_to_index) 
#reward_to_index = np.load('reward_to_index.npy', allow_pickle=True).item()
#index_to_reward = np.load('index_to_reward.npy', allow_pickle=True).item()
number_rewards = int(len(reward_to_index.keys()))

for i in range(rewards.shape[0]):
    rewards[i] = take_closest(list(reward_to_index.keys()), rewards[i])
reward_classes = np.vectorize(reward_to_index.get)(rewards)
reward_onehot = np.eye(int(len(reward_to_index.keys())))[np.array(reward_classes, dtype=np.int8)]
    

STATE_LEN = 91
ACTION_LEN = 3
    
max_train_epochs = 30

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

input_ = Input(shape=(state.shape[1],state.shape[2],))
x = Bidirectional(LSTM(128))(input_)
x = Flatten()(x)
x = Dense(12, activation='relu', name='hidden')(x)
#x = Dense(128, activation='relu', name='hidden2')(x)
out = Dense(number_rewards, activation='softmax')(x)
base_model = Model(input_, out)

base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
p = np.random.permutation(reward_onehot.shape[0])
p = np.random.permutation(reward_onehot.shape[0])
with tf.device("/device:GPU:1"):
    history = base_model.fit(state[p,:,:], reward_onehot[p], epochs=max_train_epochs, validation_split=0.5, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True,
                                    workers=4)

from sklearn.metrics import confusion_matrix

#Predict
y_prediction = base_model.predict(state)


#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(np.argmax(reward_onehot, axis=1), np.argmax(y_prediction,axis=1), normalize='pred')
np.save('reward_matrix_lstm.npy', result)