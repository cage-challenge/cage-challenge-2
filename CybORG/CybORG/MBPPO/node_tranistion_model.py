import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from keras.models import Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Input
import tensorflow as tf
from keras import backend as K

class CAGENodeTranistionModel(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(self):
        
        self.STATE_LEN = 91
        self.ACTION_LEN = 3

        self.global_itr = 0
        self.valid_split = 0.2
        self.max_train_epochs = 50

        self.input_len = 23+13+13

        #super(CAGEStateTranistionModel, self).__init__()

        losses = []
        input_ = Input(shape=(self.input_len,))
        x = Dense(128, activation='tanh', name='hidden')(input_)
        outs = []

        outs.append(Dense(3, activation='softmax', name='activity')(x))
        outs.append(Dense(4, activation='softmax', name='compromised')(x))
        losses.append(tf.keras.losses.CategoricalCrossentropy())

        self.base_model = Model(input_, outs)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        self.es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0005)
        def scheduler(epoch, lr):
            if epoch < 2:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    def forward(self, x):
        state = x[0,:91]
        #step = x[0,-1]
        action = x[0,92:]
        next_state = np.zeros(self.STATE_LEN)
        privileged = state[np.arange(3,91,step=7)]
        user = state[np.arange(4,91,step=7)]
        unknown = state[np.arange(5,91,step=7)]
        next_state = np.zeros(self.STATE_LEN)

        for i in range(13):
            encoding = np.zeros(13)
            encoding[i] = 1
            node_state = state[int(i*7):int(i*7)+7]
            node_action = self.node_action(action, i)
            probs = self.base_model(np.expand_dims([np.concatenate([encoding, node_state, node_action, privileged, unknown])], axis=-1))
            #probs = self.base_model(np.expand_dims([np.concatenate([encoding, node_state, node_action, privileged, user, unknown])], axis=-1))

            index_state = int(i*7) 
            p = probs[0].numpy()[0]
            next_state[index_state+np.random.choice(np.arange(3), p=p)] = 1

            index_state = int(i*7) + 3 
            p = probs[1].numpy()[0]
            next_state[index_state+np.random.choice(np.arange(4), p=p)] = 1
          
        return next_state
    
    def node_action(self, action, node):
        vec = np.zeros(3)
        if action < 2: 
            return vec
        action -= 2
        if action % 13 == node:
            #Analyse #Remove #Resotre
            vec[int(action) // 13] = 1
        return vec

    def fit(self, samples):

       
        K.set_value(self.base_model.optimizer.learning_rate, 0.001)
        # Process Samples
        samples = self.process_samples(samples)

        data_map = {}
        data_map['activity'] = samples['next_nodes'][:,:3]
        data_map['compromised'] = samples['next_nodes'][:,3:]
        with tf.device("/device:GPU:1"):
            history = self.base_model.fit(samples['nodes'], data_map, epochs=self.max_train_epochs, validation_split=0.1, verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=256)
        print('state tranistion val loss: ', history.history['val_loss'])

        self.global_itr += 1
        # Returns Metric Dictionary
        return self.metrics
        
    def process_samples(self, samples: SampleBatchType):
        processed = {}
        nodes = np.zeros((samples['obs'].shape[0]*13, self.input_len))
        next_nodes = np.zeros((samples['obs'].shape[0]*13, 7))
        index = 0
        for i in range(samples['obs'].shape[0]):
            step = samples['obs'][i][-1]
            privileged = samples['obs'][i][np.arange(3,91,step=7)]
            user = samples['obs'][i][np.arange(4,91,step=7)]
            unknown = samples['obs'][i][np.arange(5,91,step=7)]
            for n in range(13):
                #[encoding, step, node_state, node_action]
                encoding = np.zeros(13)
                encoding[n] = 1
                node_state = samples['obs'][i][int(n*7):int(n*7)+7]
                node_action = self.node_action(samples['actions'][i], n)
                nodes[index,:] = np.concatenate([encoding, node_state, node_action, privileged, unknown])
                #nodes[index,:] = np.concatenate([encoding, node_state, node_action, privileged, user, unknown])

                next_nodes[index,:] = samples['next_obs'][i][int(n*7):int(n*7)+7]
                index += 1

        processed['nodes'] = nodes
        processed['next_nodes'] = next_nodes
        return SampleBatch(processed)
