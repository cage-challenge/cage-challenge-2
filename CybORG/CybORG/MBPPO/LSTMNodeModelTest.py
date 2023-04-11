import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional, concatenate
#from ProcessTrueStateActionData import read_df_in_chunks
from keras.utils.vis_utils import plot_model

train_test_split = 0.75
data_path = '/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/Notebooks/logs/PPO/no_decoy_200000/data_seqence_10'
nodes = np.load(data_path + '/nodes.npy')
actions = np.load(data_path + '/actions.npy')
node_id = np.load(data_path + '/node_id.npy')
next_nodes = np.load(data_path + '/next_nodes.npy')
exploit = np.load(data_path + '/exploit.npy')
#scan = np.load(data_path + '/scan.npy')
privileged = np.load(data_path + '/privileged.npy')
user = np.load(data_path + '/user.npy')
unknown = np.load(data_path + '/unknown.npy')
#no = np.load(data_path + '/no.npy')

data = np.concatenate([nodes, actions, privileged, unknown, exploit, user], axis=-1)
print('loaded data')
    
single_node_id = np.zeros((data.shape[0],node_id.shape[2]))
for i in range(data.shape[0]):
    single_node_id[i,:] = node_id[i,0,:]

max_train_epochs = 50

losses = []
ins = []
input_ = Input(shape=(data.shape[1],data.shape[2],))
id_input = Input(13,)
x = Bidirectional(LSTM(64))(input_)
x = concatenate([x, id_input])
x = Dense(64, activation='relu', name='hidden')(x)
x = Dropout(0.2)(x)
y = Dense(32, activation='relu', name='hidden_activity')(x)
y = Dropout(0.2, name='dropout_activity')(y)
z = Dense(32, activation='relu', name='hidden_compromised')(x)
z = Dropout(0.2, name='dropout_compromised')(z)
outs = []
ins = [input_, id_input]
outs.append(Dense(3, activation='softmax', name='activity')(y))
outs.append(Dense(4, activation='softmax', name='compromised')(z))
losses.append(tf.keras.losses.CategoricalCrossentropy())
losses.append(tf.keras.losses.CategoricalCrossentropy())

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

base_model = Model(ins, outs)
base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#plot_model(base_model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

data_map = {}
p = np.random.permutation(data.shape[0])
data_map['activity'] = next_nodes[p,:3]
data_map['compromised'] = next_nodes[p,3:]
with tf.device("/device:GPU:1"):
    history = base_model.fit([data[p,:,:],single_node_id[p,:]], data_map, epochs=max_train_epochs, validation_split=0.5, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True,
                                    workers=4)

base_model.save_weights('NodeTranistionModel')