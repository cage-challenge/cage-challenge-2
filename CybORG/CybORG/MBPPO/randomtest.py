import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, Bidirectional, concatenate

losses = []
ins = []
input_ = Input(shape=(10,))
x = Dense(64, activation='relu', name='hidden')(input_)
x = Dense(64, activation='relu', name='hiddenx')(x)

outs = Dense(3, activation='softmax', name='activity')(x)

losses.append(tf.keras.losses.CategoricalCrossentropy())


def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

base_model = Model(input_, outs)
base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0005)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

x = np.zeros((500,10))
y = np.zeros((500, 3))
y[:400, 0] = 1
y[400:, 1] = 1

with tf.device("/device:GPU:0"):
    history = base_model.fit(x, y, epochs=10, callbacks=[es_callback, lr_callback], batch_size=256, shuffle=True,
                                    workers=4)
    
print(base_model.predict(x[0,:]))

