import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional, concatenate
from bisect import bisect_left
from tqdm import tqdm

STATE_LEN = 91
ACTION_LEN = 3
    
max_train_epochs = 30

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

data_path = '/home/ubuntu/u75a-Data-Efficient-Decisions/CybORG/CybORG/OfflineExperiments/logs/PPO/B_Line_no_decoy_800000/data_seqence_40'
actions_oh = np.load(data_path + '/actions.npy')[0:400000]
state = np.load(data_path + '/states.npy')[0:400000]
next_state = np.load(data_path + '/next_states.npy')[0:400000]
rewards = np.load(data_path + '/rewards.npy')[0:400000]

state_actions = np.concatenate([state, actions_oh], axis=-1)

del actions_oh
del state

input_ = Input(shape=(state_actions.shape[1],state_actions.shape[2],), name='state_action')
new_state_in = Input(shape=(next_state.shape[1]),name='state_in')

x = Bidirectional(LSTM(64))(input_)
x = Flatten()(x)
x = concatenate([x, new_state_in], name='concate')
x = Dense(128, activation='relu', name='hidden')(x)
x = Dense(32, activation='relu', name='hidden2')(x)
#out = Dense(number_rewards, activation='softmax')(x)
out = Dense(1)(x)
base_model = Model([input_, new_state_in], out)

base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanSquaredError())#loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
p = np.random.permutation(rewards.shape[0])
with tf.device("/device:GPU:1"):
    history = base_model.fit([state_actions[p,:,:], next_state[p,:]], rewards[p], epochs=max_train_epochs, validation_split=0.2, 
                                    verbose=0, callbacks=[es_callback, lr_callback], batch_size=512, shuffle=True,
                                    workers=4)

base_model.save_weights('reward_model_lstm')

from sklearn.metrics import confusion_matrix

#Predict
y_prediction = base_model.predict([state_actions[p,:,:], next_state[p,:]])
print(y_prediction.mean(axis=0))
print(rewards.mean(axis=0))

# #Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(np.argmax(reward_onehot, axis=1), np.argmax(y_prediction,axis=1), normalize='pred')
np.save('reward_matrix_lstm.npy', result)