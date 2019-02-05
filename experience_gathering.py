import numpy as np
import random
import logging
from typing import Callable
from env_2048 import Env2048, StepResult
from collections import namedtuple

main_logger = logging.getLogger('main')

Experience = namedtuple('Experience', 'from_state action reward to_state done')
PlayResults = namedtuple('PlayResults', 'experiences, scores')


# Not used
class ExperienceCollector:
    def __init__(self, model, reward_func: Callable[[StepResult], float], epsilon: float):
        self._model = model
        self._reward_func = reward_func
        self._env = Env2048()
        self.epsilon = epsilon

    def collect(self, games: int):
        experiences = []
        scores = []

        for i in range(games):
            done = False
            self._env.reset()
            from_state = self._env.state()
            while not done:
                if random.random() < self.epsilon:
                    did_move = False
                    while not did_move:
                        move = random.randint(0, 3)
                        step_result = self._env.step(move)
                        did_move = step_result.board_move_result.move_count == 0
                        done = step_result.is_game_over
                        experiences.append(
                            Experience(from_state, move, self._reward_func(step_result), step_result.state, done))
                else:
                    q_table = self._model.predict(from_state.reshape(1, 4, 4), batch_size=1)[0]
                    sorted_choices = np.argsort(q_table)
                    for move in sorted_choices:
                        step_result = self._env.step(move)
                        done = step_result.is_game_over
                        experiences.append(
                            Experience(from_state, move, self._reward_func(step_result), step_result.state, done))
                        did_move = step_result.board_move_result.move_count != 0
                        if did_move:
                            break

                from_state = step_result.state
            scores.append(self._env.score())
        return PlayResults(experiences, scores)
