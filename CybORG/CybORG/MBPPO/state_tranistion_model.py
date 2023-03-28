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

class CAGEStateTranistionModel(TFModelV2):
    """Transition Dynamics Model (FC Network with Weight Norm)"""

    def __init__(self):
        
        self.NUM_NODES = 13
        self.NODE_CLASSES = [3, 4]
        self.STATE_LEN = 91
        self.ACTION_LEN = 41

        self.global_itr = 0
        self.valid_split = 0.2
        self.max_train_epochs = 50

        #super(CAGEStateTranistionModel, self).__init__()

        losses = []
        input_ = Input(shape=(self.STATE_LEN+self.ACTION_LEN+1,))
        outs = []
        for i in range(self.NUM_NODES):
            for n in self.NODE_CLASSES:
                #x_ = Dense(128, activation='relu')(input_)
                outs.append(Dense(n, activation='softmax', name=str(i)+str(n))(input_))
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
        probs = self.base_model(x)
        next_state = np.zeros(self.STATE_LEN)
        index_state = 0; index = 0
        for i in range(self.NUM_NODES):
            for n in self.NODE_CLASSES:
                #p = probs[index][0]
                #p /= p.sum()
                #next_state[index_state+np.random.choice(np.arange(n), p=p)] = 1
                #next_state[index_state+tf.random.categorical(probs[index], 1)[0][0]] = 1
                #index_state += n; index += 1
                p = probs[index].numpy()[0]
                next_state[index_state+np.random.choice(np.arange(n), p=p)] = 1
                index_state += n; index += 1
        return next_state
    

    def fit(self, samples):

       
        K.set_value(self.base_model.optimizer.learning_rate, 0.001)
        # Process Samples
        samples = self.process_samples(samples)

        data_map = {}
        index = 0 
        for i in range(self.NUM_NODES):
            for n in self.NODE_CLASSES:
                data_map[str(i)+str(n)] = samples['next_obs'][:,index:index+n]
                index += n

        history = self.base_model.fit(samples['obs_actions'], data_map, epochs=self.max_train_epochs, validation_split=0.1, verbose=0, callbacks=[self.es_callback, self.lr_callback], batch_size=256)
        print('state tranistion val loss: ', history.history['val_loss'])
        
        K.clear_session()

        self.global_itr += 1
        # Returns Metric Dictionary
        return self.metrics
        
    def process_samples(self, samples: SampleBatchType):
        processed = {}
        actions_onehot = np.zeros((samples['obs'].shape[0], self.ACTION_LEN))
        actions_onehot[np.arange(samples['obs'].shape[0], dtype=np.int32),np.array(samples['actions'], np.int16)] = 1  
        processed['obs_actions'] = np.concatenate([samples['obs'], actions_onehot], axis=1)
        processed['next_obs'] = samples['next_obs']
        return SampleBatch(processed)
