from utils.logger import Logger


class BaseModel:
    def __init__(self, game_name, mode_name, log_directory, input_shape, action_space):
        self.logger = Logger('%s_%s' % (game_name, mode_name), log_directory)
        self.input_shape = input_shape
        self.action_space = action_space

    def step_update(self, total_step):
        pass

    def save_run(self, score, step):
        self.logger.add_score(score)
        self.logger.add_step(step)

    def move(self, state):
        pass
