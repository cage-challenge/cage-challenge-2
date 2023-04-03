import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input
#from ProcessTrueStateActionData import read_df_in_chunks


train_test_split = 0.75
data_path = '/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/Notebooks/logs/PPO/no_decoy_200000'
nodes = np.load(data_path + '/data/nodes.npy')
actions = np.load(data_path + '/data/actions.npy')
node_id = np.load(data_path + '/data/node_id.npy')
next_nodes = np.load(data_path + '/data/next_nodes.npy')
exploit = np.load(data_path + '/data/exploit.npy')
scan = np.load(data_path + '/data/scan.npy')
privileged = np.load(data_path + '/data/privileged.npy')
user = np.load(data_path + '/data/user.npy')
unknown = np.load(data_path + '/data/unknown.npy')
no = np.load(data_path + '/data/no.npy')

#data = np.concatenate([node_id, nodes, actions, privileged, unknown, exploit, user], axis=1)
data = np.concatenate([node_id, nodes, actions, privileged, unknown], axis=1)
    
max_train_epochs = 50
seq_len = 5

losses = []
input_ = Input(shape=(seq_len, data.shape[1],))
x = Dense(64, activation='relu', name='hidden')(input_)
outs = []

outs.append(Dense(3, activation='softmax', name='activity')(x))
outs.append(Dense(4, activation='softmax', name='compromised')(x))
losses.append(tf.keras.losses.CategoricalCrossentropy())
losses.append(tf.keras.losses.CategoricalCrossentropy())

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.15)

base_model = Model(input_, outs)
base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0005)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

data_map = {}
p = np.random.permutation(data.shape[0])
data_map['activity'] = next_nodes[p,:3]
data_map['compromised'] = next_nodes[p,3:]
with tf.device("/device:GPU:1"):
    history = base_model.fit(data[p,:], data_map, epochs=max_train_epochs, validation_split=0.5, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True)
