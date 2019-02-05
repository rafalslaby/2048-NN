from board import Board
from typing import Optional
from collections import namedtuple
from gui_2048 import GUI2048
import queue
import threading
import numpy as np
from helper_functions import game_power_2

RENDER = False

StepResult = namedtuple("StepResult", "state board_move_result is_game_over")
Score = namedtuple('Score', 'max_tile sum')


class Env2048:
    GAME_OVER_SLEEP = 3
    RENDER_QUEUE_SIZE = 100

    def __init__(self, board: Optional[Board] = None, gui_visualization=False, render_fps=None):
        self._board = board or Board()
        self._bases = np.ones((4, 4), 'int32')
        self._do_visualize = gui_visualization
        self.render_fps = render_fps
        if gui_visualization:
            self.state_render_queue = queue.Queue(maxsize=Env2048.RENDER_QUEUE_SIZE)
            self.game_thread = threading.Thread(target=GUI2048, args=(self.state_render_queue, 1000 // render_fps))
            self.game_thread.start()
            self._render()

    def reset(self):
        self._board = Board()

    def step(self, action, render=True):
        render = render and self._do_visualize
        board_move_result = self._board.make_move(action)
        if render:
            self._render()
        if board_move_result.move_count > 0:
            self._board.add_random()
            if render:
                self._render()
        is_game_over = self._is_game_over()
        if render and is_game_over:
            self._render(Env2048.GAME_OVER_SLEEP * 1000)
        step_result = StepResult(self.state(), board_move_result, is_game_over)
        return step_result

    def dry_step(self, action):
        board_backup = self._board.state.copy()
        result = self.step(action, render=False)
        self._board.state = board_backup
        return result

    def state(self):
        return self._board.state.copy()

    def power_2_state(self):
        return game_power_2(self.state())

    def score(self):
        return Score(self._board.max_number(), self._board.sum())

    def act_space(self):
        return self._board.available_moves()

    def _is_game_over(self):
        return self._board.is_board_full() and not self._board.can_move()

    def _render(self, wait_ms=None):
        self.state_render_queue.put((self.power_2_state(), wait_ms))
