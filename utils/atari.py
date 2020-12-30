from models.ddqn_model import DdqnSolver, DdqnTrainer
import gym
from .atari_wrappers import NoopResetEnv, FireResetEnv, WarpFrame, FrameStack
from os import path, getcwd, getenv
import numpy as np
from dotenv import load_dotenv
load_dotenv(path.join(getcwd(), '.env'))

FRAMES_IN_OBSERVATION = int(getenv('FRAMES_IN_OBSERVATION'))
FRAME_SIZE = int(getenv('FRAME_SIZE'))
INPUT_SHAPE = (FRAME_SIZE, FRAME_SIZE, FRAMES_IN_OBSERVATION)


class Atari:
    def __init__(self, game_name, mode_name='training', total_step_limit=5000000, total_run_limit=None, should_render=True, sign_only=False):
        env_name = game_name + 'NoFrameskip-v4'
        self.__env = Atari.__generate_env(env_name)
        self.__env.seed(42)
        self.__game_model = Atari.__generate_model(
            game_name, mode_name, self.__env.action_space.n)
        self.__should_render = should_render
        self.__total_step_limit = total_step_limit
        self.__total_run_limit = total_run_limit
        self.__sign_only = sign_only

    def loop(self):
        total_run = 0
        total_step = 0
        while True:
            if (self.__total_run_limit is not None and total_run >= self.__total_run_limit):
                print('Reached total run limit')
                return
            total_run += 1
            current_state = self.__env.reset()
            score = 0
            step = 0
            while True:
                if (self.__total_step_limit is not None and total_step >= self.__total_step_limit):
                    print('Reached total step limit')
                    return
                total_step += 1
                step += 1

                if (self.__should_render):
                    self.__env.render()

                action = self.__game_model.move(current_state)
                next_state, reward, done, _ = self.__env.step(action)
                if (done):
                    reward = -1
                if (self.__sign_only):
                    reward = np.sign(reward)
                score += reward
                self.__game_model.remember(
                    current_state, action, reward, next_state, done)
                current_state = next_state

                self.__game_model.step_update(total_step)
                if (done):
                    self.__game_model.save_run(score, step)
                    break

    def save(self):
        self.__game_model.save_weights()

    @staticmethod
    def __generate_model(game_name, mode_name, action_space):
        game_directory = Atari.__get_game_directory(game_name)
        model_path = path.join(game_directory, 'model.h5')
        if (mode_name == 'training'):
            data_path = path.join(game_directory, 'data.pkl')
            return DdqnTrainer(game_name, path.join(game_directory, 'logs'), INPUT_SHAPE, action_space, model_path, data_path)
        if (mode_name == 'testing'):
            return DdqnSolver(game_name, path.join(game_directory, 'logs'), INPUT_SHAPE, action_space, model_path)
        raise RuntimeError('Unrecognized mode_name: %s' % mode_name)

    @staticmethod
    def __get_game_directory(game_name):
        return path.join(getcwd(), 'games', game_name)

    @staticmethod
    def __generate_env(env_name):
        env = gym.make(env_name)
        env = NoopResetEnv(env)
        if ('FIRE' in env.unwrapped.get_action_meanings()):
            env = FireResetEnv(env)
        env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)
        env = FrameStack(env, k=FRAMES_IN_OBSERVATION)
        return env
