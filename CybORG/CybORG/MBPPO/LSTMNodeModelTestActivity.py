import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
actions_oh = np.load(data_path + '/actions_oh.npy')
#no = np.load(data_path + '/no.npy')

state = np.load(data_path + '/states.npy')
state = np.array(state, dtype=np.int8)
state = np.repeat(state, 13, axis=0)
actions_oh = np.array(actions_oh, dtype=np.int8)
actions_oh = np.repeat(actions_oh, 13, axis=0)
data = np.concatenate([state, actions_oh], axis=-1)
#data = np.concatenate([nodes, actions, exploit, privileged, user, unknown], axis=-1)

del nodes
del actions
del exploit
del privileged
del user
del unknown
del state
del actions_oh

#data = np.concatenate([state, actions], axis=-1)
print('loaded data')

predictions = np.zeros((data.shape[0],91))
for i in range(data.shape[0]):
    index = (i // 13) * 13
    range_ = i % 13
    if range_ > 0:
        predictions[i,:int(range_*7)] = next_nodes[index:index+range_,:].reshape(-1)
    

single_node_id = np.zeros((data.shape[0],node_id.shape[2]))
for i in range(data.shape[0]):
    single_node_id[i,:] = node_id[i,0,:]

max_train_epochs = 50
print(data.shape[2])
losses = []
ins = []
input_ = Input(shape=(data.shape[1],data.shape[2],))
id_input = Input(13,)
pred = Input(91,)
x = Bidirectional(LSTM(64))(input_)
x = concatenate([x, id_input, pred])
x = Dense(64, activation='relu', name='hidden')(x)
x = Dropout(0.2)(x)
z = Dense(32, activation='relu', name='hidden_activity')(x)
z = Dropout(0.2, name='dropout_activity')(z)
outs = []
ins = [input_, id_input, pred]
outs.append(Dense(3, activation='softmax', name='activity')(z))
losses.append(tf.keras.losses.CategoricalCrossentropy())

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.03)

base_model = Model(ins, outs)
base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#plot_model(base_model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

p = np.random.permutation(data.shape[0])

with tf.device("/device:CPU:30"):
    history = base_model.fit([data[p,:,:],single_node_id[p,:], predictions[p,:]], next_nodes[p,:3], epochs=max_train_epochs, validation_split=0.5, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True,
                                    workers=4)

base_model.save_weights('NodeTranistionModelFA')

r = base_model.predict([data[p,:,:],single_node_id[p,:], predictions[p,:]], batch_size=1024)

print('From data: ', next_nodes[p,:3].mean(axis=0))
print('pred: ', r.mean(axis=0))
s = np.zeros((10000, 4))
for i in range(10000):
    s[i,np.random.choice(np.arange(3), p=r[i,:])] = 1

print('pred sampled: ', s.mean(axis=0))