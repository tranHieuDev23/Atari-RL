from .base_model import BaseModel
from .ddqn_cnn import get_network
from random import random, randrange
from collections import deque
from os import path, getenv, getcwd
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
import keras
load_dotenv(path.join(getcwd(), '.env'))

GAMMA = float(getenv('GAMMA'))
MEMORY_SIZE = int(getenv('MEMORY_SIZE'))
MEMORY_REPLAY_SIZE = int(getenv('MEMORY_REPLAY_SIZE'))
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
        self.model.summary()
        self.__target_model = get_network(input_shape, action_space)
        self.__reset_target_model()
        self.__loss_function = keras.losses.Huber()
        self.__optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

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
        self.total_step = current_step
        self.__update_epsilon()

        if (self.total_step % TRAINING_FREQUENCY == 0 and len(self.memory) >= MEMORY_REPLAY_SIZE):
            loss, average_max_q = self.__train()
            self.logger.add_loss(loss)
            self.logger.add_q(average_max_q)

        if (self.total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0):
            self.save()

        if (self.total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0):
            self.__reset_target_model()

        if (self.total_step % LOG_FREQUENCY == 0):
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(self.total_step))

    def save(self):
        self.save_weights()

    def __train(self):
        indices = np.random.choice(range(len(self.memory)), size=MEMORY_REPLAY_SIZE)
        states = np.asarray([np.asarray(self.memory[i]['state']) for i in indices])
        next_states = np.asarray([np.asarray(self.memory[i]['next_state']) for i in indices])
        dones = tf.convert_to_tensor(
            [float(self.memory[i]['done']) for i in indices]
        )
        rewards = np.asarray([np.asarray(self.memory[i]['reward']) for i in indices])
        actions = np.asarray([np.asarray(self.memory[i]['action']) for i in indices])

        next_q_values = self.__target_model.predict(next_states, verbose=0, batch_size=BATCH_SIZE)
        updated_q_values = rewards + GAMMA * np.max(next_q_values, axis=1)
        updated_q_values = updated_q_values * (1 - dones) - dones

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            masks = tf.one_hot(actions, self.action_space)
            q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.__loss_function(updated_q_values, q_actions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.__optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(loss), np.mean(np.max(q_values, axis=1))

    def __update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(self.epsilon, EXPLORATION_MIN)

    def __reset_target_model(self):
        self.__target_model.set_weights(self.model.get_weights())
