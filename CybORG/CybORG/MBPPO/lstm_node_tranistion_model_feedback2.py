import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import time
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, LSTM, Input, concatenate
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import backend as K
from numba import cuda
import joblib
from tqdm.keras import TqdmCallback

class CAGENodeTranistionModelLSTMFeedback2(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(self, seq_len):
        
        self.STATE_LEN = 91
        self.ACTION_LEN = 41
        self.SEQ_LEN = seq_len
        self.global_itr = 0
        self.valid_split = 0.2
        self.max_train_epochs = 50
        self.input_len = self.STATE_LEN + self.ACTION_LEN

        #super(CAGEStateTranistionModel, self).__init__()

        losses = []
        input_ = Input(shape=(self.SEQ_LEN,self.input_len,), name='main_in')
        id_input = Input(13, name='id_in')
        prediction = Input(91, name='pred_in')
        x = Bidirectional(LSTM(64),name='lstm')(input_)
        x = concatenate([x, id_input, prediction],name='concate')
        x = Dense(64, activation='relu', name='hidden')(x)
        x = Dropout(0.2)(x)
        y = Dense(32, activation='relu', name='hidden_activity')(x)
        y = Dropout(0.2, name='dropout_activity')(y)

        ins = [input_, id_input, prediction]
        out1 = Dense(3, activation='softmax', name='activity')(y)

        self.base_model = Model(ins, out1)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()], run_eagerly=True)

        losses2 = []
        input_2 = Input(shape=(self.SEQ_LEN,self.input_len,))
        id_input2 = Input(13,)
        pred2 = Input(91,)
        x2 = Bidirectional(LSTM(64))(input_2)
        x2 = concatenate([x2, id_input2, pred2])
        x2 = Dense(64, activation='relu', name='hidden')(x2)
        x2 = Dropout(0.2)(x2)
        z2 = Dense(32, activation='relu', name='hidden_compromised')(x2)
        z2 = Dropout(0.2, name='dropout_compromised')(z2)
        ins2 = [input_2, id_input2, pred2]
        out2 = Dense(4, activation='softmax', name='compromised')(z2)

        self.compromised_model = Model(ins2, out2)
        self.compromised_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()], run_eagerly=True)
        
        self.es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, min_delta=0.00005, restore_best_weights=True)
        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.05)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        

    def forward(self, state, actions):
        next_state = np.zeros(self.STATE_LEN)
        # exploit = state[:,np.arange(0,91,step=7)]
        # privileged = state[:,np.arange(3,91,step=7)]
        # user = state[:,np.arange(4,91,step=7)]
        # unknown = state[:,np.arange(5,91,step=7)]
        #no = state[:,np.arange(6,91,step=7)]
        valid = -3
        while valid < 0:
            index_state = 0
            next_state = np.zeros(self.STATE_LEN)
            for n in range(13):

                encoding = np.zeros(13)
                encoding[n] = 1
                #node_state = state[:,int(n*7):int(n*7)+7]
                #node_action =  np.array([self.node_action(actions[i], n) for i in range(self.SEQ_LEN)])
            
                probs = self.base_model([np.expand_dims(np.concatenate([state, actions], axis=-1), axis=0), np.expand_dims(encoding, axis=0), np.expand_dims(next_state, axis=0)])
                #probs = self.base_model([np.expand_dims(np.concatenate([node_state, node_action, exploit, privileged, user, unknown], axis=-1), axis=0), encoding, np.expand_dims(next_state, axis=0)])

                p = probs.numpy()[0]
                next_state[index_state+np.random.choice(np.arange(3), p=p)] = 1
                index_state += 3

                probs = self.compromised_model([np.expand_dims(np.concatenate([state, actions], axis=-1), axis=0), np.expand_dims(encoding, axis=0), np.expand_dims(next_state, axis=0)])
                #probs = self.compromised_model([np.expand_dims(np.concatenate([node_state, node_action], axis=-1), axis=0), encoding, np.expand_dims(next_state, axis=0)])
                p = probs.numpy()[0] 
                #next_state[index_state+np.random.choice(np.arange(4), p=p)] = 1
                next_state[index_state+np.argmax(p)] = 1
                index_state += 4

            if any((next_state==self.known_states).all(1)):
                valid = 1
            else:
                valid += 1 
            
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

    def fit(self, node_ids, node_vectors, predictions1, predictions2, next_nodes):

        K.set_value(self.base_model.optimizer.learning_rate, 0.0005)
        K.set_value(self.compromised_model.optimizer.learning_rate, 0.0005)

        p = np.random.permutation(node_vectors.shape[0])
        print('Activity From data: ', next_nodes[:,:3].mean(axis=0))
        try:
            with tf.device("/device:GPU:0"):
                history = self.base_model.fit([node_vectors[p,:,:],node_ids[p,:],predictions1[p,:]], next_nodes[p,:3], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                            verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=1024, shuffle=True, workers=4)
            print('Activity val loss: ', history.history['val_loss'])
            print('Activity val accuracy ', history.history['val_categorical_accuracy'])
        except: #Memory allocation issues
            time.sleep(15)
            try:
                with tf.device("/device:CPU:38"):
                    history = self.base_model.fit([node_vectors[p,:,:],node_ids[p,:],predictions2[p,:]], next_nodes[p,:3], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                                verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=1024, shuffle=True, workers=4)
                    print('Compromised val loss: ', history.history['val_loss'])
                    print('Compromised val accuracy ', history.history['val_categorical_accuracy'])
            except:
                print('Activity train fail')

        K.clear_session()
        print('Compromised From data: ', next_nodes[:,3:].mean(axis=0))
        try:
            with tf.device("/device:GPU:1"):
                history = self.compromised_model.fit([node_vectors[p,:,:],node_ids[p,:],predictions2[p,:]], next_nodes[p,3:], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                            verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=1024, shuffle=True, workers=4)
            print('Compromised val loss: ', history.history['val_loss'])
            print('Compromised val accuracy ', history.history['val_categorical_accuracy'])
        except: #Memory allocation issues
            time.sleep(15)
            try:
                with tf.device("/device:CPU:38"):
                    history = self.compromised_model.fit([node_vectors[p,:,:],node_ids[p,:],predictions2[p,:]], next_nodes[p,3:], epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                                verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=1024, shuffle=True, workers=4)
                    print('Compromised val loss: ', history.history['val_loss'])
                    print('Compromised val accuracy ', history.history['val_categorical_accuracy'])
            except:
                print('Compromised train fail')
        K.clear_session()
        self.global_itr += 1
        # Returns Metric Dictionary

        return self.metrics
    
    def load(self, path):
        self.base_model.load_weights(path)

    def load_comp(self, path):
        self.compromised_model.load_weights(path)

    def get_weights(self):
        return [self.base_model.get_weights(), self.compromised_model.get_weights()]
    
    def set_weights(self, weights, unique):
        self.base_model.set_weights(weights[0])
        self.compromised_model.set_weights(weights[1])
        self.known_states = unique
        
   