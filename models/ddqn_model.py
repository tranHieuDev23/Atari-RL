from .base_model import BaseModel
from .ddqn_cnn import get_network
from random import random, randrange, sample
from collections import deque
from statistics import mean
from os import path, getenv, getcwd
import numpy as np
from dotenv import load_dotenv
import pickle as pkl
load_dotenv(path.join(getcwd(), '.env'))

GAMMA = float(getenv('GAMMA'))
MEMORY_SIZE = int(getenv('MEMORY_SIZE'))
BATCH_SIZE = int(getenv('BATCH_SIZE'))
TRAINING_FREQUENCY = int(getenv('TRAINING_FREQUENCY'))
TARGET_NETWORK_UPDATE_FREQUENCY = int(
    getenv('TARGET_NETWORK_UPDATE_FREQUENCY'))
MODEL_PERSISTENCE_UPDATE_FREQUENCY = int(
    getenv('MODEL_PERSISTENCE_UPDATE_FREQUENCY'))
LOG_FREQUENCY = int(getenv('LOG_FREQUENCY'))
REPLAY_START_SIZE = int(getenv('REPLAY_START_SIZE'))

EXPLORATION_MAX = float(getenv('EXPLORATION_MAX'))
EXPLORATION_MIN = float(getenv('EXPLORATION_MIN'))
EXPLORATION_TEST = float(getenv('EXPLORATION_TEST'))
EXPLORATION_STEPS = int(getenv('EXPLORATION_STEPS'))
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


class DdqnModel(BaseModel):
    def __init__(self, game_name, mode_name, log_directory, input_shape, action_space, model_path, start_epsilon):
        super().__init__(game_name, mode_name, log_directory, input_shape, action_space)
        if (model_path is None):
            raise RuntimeError('model_path should not be None')
        self.model_path = model_path
        self.model = get_network(input_shape, action_space)
        if (path.isfile(model_path)):
            self.model.load_weights(model_path)
        self.epsilon = start_epsilon

    def save_model(self):
        if (self.model_path is None):
            raise RuntimeError('model_path is None')
        self.model.save(self.model_path)

    def save_weights(self):
        if (self.model_path is None):
            raise RuntimeError('model_path is None')
        self.model.save_weights(self.model_path)

    def remember(self, state, action, reward, next_state, done):
        pass


class DdqnSolver(DdqnModel):
    def __init__(self, game_name, log_directory, input_shape, action_space, model_path):
        super().__init__(game_name, 'Testing', log_directory,
                         input_shape, action_space, model_path, EXPLORATION_TEST)

    def move(self, state):
        if (random() < self.epsilon):
            return randrange(self.action_space)
        q_hat = self.model.predict(np.expand_dims(
            np.asarray(state).astype(np.float64), axis=0))
        return np.argmax(q_hat)


class DdqnTrainer(DdqnModel):
    def __init__(self, game_name, log_directory, input_shape, action_space, model_path, data_path):
        super().__init__(game_name, 'Training', log_directory,
                         input_shape, action_space, model_path, EXPLORATION_MAX)
        self.total_step = 0
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.data_path = data_path
        if (path.isfile(data_path)):
            with open(data_path, 'rb') as data_file:
                data = pkl.load(data_file)
                self.total_step = data['total_step']
                self.epsilon = data['epsilon']
                self.memory = data['memory']

        self.model.summary()
        self.__target_model = get_network(input_shape, action_space)
        self.__reset_target_model()

    def move(self, state):
        if (random() < self.epsilon or len(self.memory) < REPLAY_START_SIZE):
            return randrange(self.action_space)
        q_hat = self.model.predict(np.expand_dims(
            np.asarray(state).astype(np.float64), axis=0))
        return np.argmax(q_hat)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

    def step_update(self, current_step):
        self.total_step += 1
        if (len(self.memory) < REPLAY_START_SIZE):
            return

        if (self.total_step % TRAINING_FREQUENCY == 0):
            loss, average_max_q = self.__train()
            self.logger.add_loss(loss)
            self.logger.add_q(average_max_q)
            print('{{"metric": "loss", "value": {}}}'.format(loss))
            print('{{"metric": "average_max_q", "value": {}}}'.format(average_max_q))

        self.__update_epsilon()

        if (self.total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0):
            self.save()

        if (self.total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0):
            self.__reset_target_model()

        if (self.total_step % LOG_FREQUENCY == 0):
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(self.total_step))

    def save(self):
        self.save_weights()
        data = {
            'total_step': self.total_step,
            'epsilon': self.epsilon,
            'memory': self.memory
        }
        with open(self.data_path, 'wb') as data_file:
            pkl.dump(data, data_file)

    def __train(self):
        batch = np.asarray(sample(self.memory, BATCH_SIZE))
        if (len(batch) < BATCH_SIZE):
            return

        states = []
        q_values = []
        max_q_values = []
        for item in batch:
            state = np.asarray(item['state']).astype(np.float64)
            next_state = np.asarray(item['next_state']).astype(np.float64)
            q_value = self.model.predict(
                np.expand_dims(state, 0), verbose=0)[0]
            next_q_value = np.max(
                self.__target_model.predict(np.expand_dims(next_state, 0), verbose=0)[0])
            if (item['done']):
                q_value[item['action']] = item['reward']
            else:
                q_value[item['action']] = item['reward'] + GAMMA * next_q_value
            states.append(state)
            q_values.append(q_value)
            max_q_values.append(np.max(q_value))

        fit = self.model.fit(x=np.asarray(states), y=np.asarray(
            q_values), batch_size=BATCH_SIZE)
        loss = fit.history['loss'][0]
        return loss, mean(max_q_values)

    def __update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(self.epsilon, EXPLORATION_MIN)

    def __reset_target_model(self):
        self.__target_model.set_weights(self.model.get_weights())
