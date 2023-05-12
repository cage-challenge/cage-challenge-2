
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["SM_FRAMEWORK"] = "tf.keras"

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
        
x = np.zeros((100, 10))
y = np.zeros((100, 10))
for i in range(100):
    index = i % 10
    y[i, index] = 1

input_ = Input(shape=(10,))
x = Dense(32, activation='relu', name='hidden2')(input_)
#out = Dense(number_rewards, activation='softmax')(x)
out = Dense(10)(x)
base_model = Model(input_, out)

base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanSquaredError())#loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
with tf.device("/device:CPU:1"):
    history = base_model.fit(x, y, epochs=max_train_epochs, validation_split=0.0, 
                                    verbose=2, callbacks=[es_callback, lr_callback], batch_size=100, shuffle=True,
                                    workers=4)
    
print(base_model.predict(np.zeros((1, 10))))