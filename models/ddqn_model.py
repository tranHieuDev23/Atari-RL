from .base_model import BaseModel
from .ddqn_cnn import get_network
from random import random, randrange, sample
from collections import deque
from statistics import mean
from os import path, getenv, getcwd
import numpy as np
from dotenv import load_dotenv
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
    def __init__(self, game_name, mode_name, log_directory, input_shape, action_space, model_path):
        super().__init__(game_name, mode_name, log_directory, input_shape, action_space)
        if (model_path is None):
            raise RuntimeError('model_path should not be None')
        self.model_path = model_path
        self.model = get_network(input_shape, action_space)
        if (path.isfile(model_path)):
            self.model.load_weights(model_path)

    def save_model(self):
        if (self.model_path is None):
            raise RuntimeError('model_path is None')
        self.model.save(self.model_path)

    def save_weights(self):
        if (self.model_path is None):
            raise RuntimeError('model_path is None')
        self.model.save_weights(self.model_path)


class DdqnSolver(DdqnModel):
    def __init__(self, game_name, log_directory, input_shape, action_space, model_path):
        super().__init__(game_name, 'Testing', log_directory,
                         input_shape, action_space, model_path)

    def move(self, state):
        if (random() < EXPLORATION_TEST):
            return randrange(self.action_space)
        q_hat = self.model.predict(np.expand_dims(
            np.asarray(state).astype(np.float64), axis=0))
        return np.argmax(q_hat)


class DdqnTrainer(DdqnModel):
    def __init__(self, game_name, log_directory, input_shape, action_space, model_path):
        super().__init__(game_name, 'Training', log_directory,
                         input_shape, action_space, model_path)
        self.model.summary()
        self.__target_model = get_network(input_shape, action_space)
        self.__reset_target_model()
        self.epsilon = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)

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

    def step_update(self, total_step):
        if (len(self.memory) < REPLAY_START_SIZE):
            return

        if (total_step % TRAINING_FREQUENCY == 0):
            loss, accuracy, average_max_q = self.__train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)
            print('{{"metric": "loss", "value": {}}}'.format(loss))
            print('{{"metric": "accuracy", "value": {}}}'.format(accuracy))
            print('{{"metric": "average_max_q", "value": {}}}'.format(average_max_q))

        self.__update_epsilon()

        if (total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0):
            self.save_weights()

        if (total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0):
            self.__reset_target_model()

        if (total_step % LOG_FREQUENCY == 0):
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def __train(self):
        batch = np.asarray(sample(self.memory, BATCH_SIZE))
        if (len(batch) < BATCH_SIZE):
            return

        states = []
        q_values = []
        max_q_values = []
        for item in batch:
            state = np.expand_dims(np.asarray(
                item['state']).astype(np.float64), axis=0)
            next_state = np.expand_dims(np.asarray(
                item['next_state']).astype(np.float64), axis=0)
            q_value = list(self.model.predict(state, verbose=0)[0])
            next_q_value = np.max(
                self.__target_model.predict(next_state, verbose=0).ravel())
            if (item['done']):
                q_value[item['action']] = item['reward']
            else:
                q_value[item['action']] = item['reward'] + GAMMA * next_q_value

            states.append(state)
            q_values.append(q_value)
            max_q_values.append(np.max(q_value))

        fit = self.model.fit(x=np.asarray(states).squeeze(), y=np.asarray(
            q_values).squeeze(), batch_size=BATCH_SIZE, verbose=0)
        loss = fit.history['loss'][0]
        accuracy = fit.history['accuracy'][0]
        return loss, accuracy, mean(max_q_values)

    def __update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(self.epsilon, EXPLORATION_MIN)

    def __reset_target_model(self):
        self.__target_model.set_weights(self.model.get_weights())
