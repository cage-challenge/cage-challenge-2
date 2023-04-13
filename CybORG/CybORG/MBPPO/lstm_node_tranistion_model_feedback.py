import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

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
import joblib

class CAGENodeTranistionModelLSTMFeedback(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(self):
        
        self.STATE_LEN = 91
        self.ACTION_LEN = 3
        self.SEQ_LEN = 10
        self.global_itr = 0
        self.valid_split = 0.3
        self.max_train_epochs = 50

        self.input_len = 7+3+13+13+13+13

        #super(CAGEStateTranistionModel, self).__init__()

        losses = []
        input_ = Input(shape=(self.SEQ_LEN,self.input_len,))
        id_input = Input(13,)
        prediction = Input(91,)
        x = Bidirectional(LSTM(64))(input_)
        x = concatenate([x, id_input, prediction])
        x = Dense(128, activation='relu', name='hidden')(x)
        x = Dropout(0.2)(x)
        y = Dense(32, activation='relu', name='hidden_activity')(x)
        y = Dropout(0.2, name='dropout_activity')(y)
        z = Dense(32, activation='relu', name='hidden_compromised')(x)
        z = Dropout(0.2, name='dropout_compromised')(z)
        ins = [id_input,input_,prediction]
        outs = []
        outs.append(Dense(3, activation='softmax', name='activity')(y))
        outs.append(Dense(4, activation='softmax', name='compromised')(z))
        losses.append(tf.keras.losses.CategoricalCrossentropy())
        losses.append(tf.keras.losses.CategoricalCrossentropy())

        self.base_model = Model(ins, outs)
        self.base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=[tf.keras.metrics.CategoricalAccuracy()], run_eagerly=True)
        
        self.es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.05)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.clf = joblib.load('/home/adamprice/u75a-Data-Efficient-Decisions/CybORG/CybORG/MBPPO/state_novelty.pkl')

    def forward(self, state, actions):
        next_state = np.zeros(self.STATE_LEN)
        exploit = state[:,np.arange(0,91,step=7)]
        privileged = state[:,np.arange(3,91,step=7)]
        user = state[:,np.arange(4,91,step=7)]
        unknown = state[:,np.arange(5,91,step=7)]
        no = state[:,np.arange(6,91,step=7)]
        next_state = np.zeros(self.STATE_LEN)

        #valid = -2
        #while valid < 0:
        for n in range(13):

            encoding = np.zeros((1,13))
            encoding[:,n] = 1
            node_state = state[:,int(n*7):int(n*7)+7]
            node_action =  np.array([self.node_action(actions[i], n) for i in range(self.SEQ_LEN)])
        
            probs = self.base_model([encoding,np.expand_dims(np.concatenate([node_state, node_action, exploit, privileged, user, unknown], axis=-1), axis=0), np.expand_dims(next_state, axis=0)])
            #probs = self.base_model(np.expand_dims(np.concatenate([encoding, node_state, node_action, exploit, privileged, user, unknown], axis=-1), axis=0))

            index_state = int(n*7) 
            p = probs[0].numpy()[0]
            next_state[index_state+np.random.choice(np.arange(3), p=p)] = 1

            index_state = int(n*7) + 3 
            p = probs[1].numpy()[0] 

            next_state[index_state+np.random.choice(np.arange(4), p=p)] = 1

                #if self.clf.predict(np.expand_dims(next_state, axis=0))[0] > 0:
                #    valid = 1
                #else:
                #    valid += 1 
            
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

    def fit(self, node_ids, node_vectors, predictions, next_nodes):

        K.set_value(self.base_model.optimizer.learning_rate, 0.0005)
        # Process Samples
        #samples = self.process_samples(obs, actions, next_obs)
        p = np.random.permutation(node_vectors.shape[0])
        data_map = {}
        data_map['activity'] = next_nodes[p,:3]
        data_map['compromised'] = next_nodes[p,3:]
        with tf.device("/device:GPU:1"):
            history = self.base_model.fit([node_ids[p,:],node_vectors[p,:,:],predictions[p,:]], data_map, epochs=self.max_train_epochs, validation_split=self.valid_split, 
                                          verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=256, shuffle=True, workers=4)
        print('state tranistion val loss: ', history.history['val_loss'])

        self.global_itr += 1
        # Returns Metric Dictionary
        return self.metrics
    
    def load(self, path):
        self.base_model.load_weights(path)
        
    def process_samples(self, obs, actions, next_obs):
        processed = {}
        nodes = np.zeros((obs.shape[0]*13, self.SEQ_LEN, self.input_len))
        next_nodes = np.zeros((obs.shape[0]*13, 7))
        index = 0

        for i in range(obs.shape[0]):
            step = obs[i][-1]
            exploit = obs[i][:,np.arange(0,91,step=7)]
            scan = obs[i][:,np.arange(0,91,step=7)]
            privileged = obs[i][:,np.arange(3,91,step=7)]
            user = obs[i][:,np.arange(4,91,step=7)]
            unknown = obs[i][:,np.arange(5,91,step=7)]
            no = obs[i][:,np.arange(6,91,step=7)]
            for n in range(13):
                encoding = np.zeros((self.SEQ_LEN, 13))
                encoding[:,n] = 1
                node_state = obs[i,:,int(n*7):int(n*7)+7]
                node_action = np.array([self.node_action(actions[i][k], n) for k in range(self.SEQ_LEN)])

                nodes[index,:] = np.concatenate([encoding, node_state, node_action, privileged, unknown], axis=-1)
                #nodes[index,:] = np.concatenate([encoding, node_state, node_action, privileged, user, unknown, no])

                next_nodes[index,:] = next_obs[i][int(n*7):int(n*7)+7]
                index += 1

        processed['nodes'] = nodes
        processed['next_nodes'] = next_nodes
        return SampleBatch(processed)
