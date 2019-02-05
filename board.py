import numpy as np
import random
from typing import Optional
from collections import namedtuple

BoardMoveResult = namedtuple("BoardMoveResult", "move_count merged_values")


class Board:
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def __init__(self, n: int = 4, initial_values: Optional[np.array] = None):
        if initial_values is not None:
            n = initial_values.shape[0]
            assert (n, n) == initial_values.shape, "Provide nxn board"
            self.state = initial_values
        else:
            self.state = np.zeros((n, n), dtype='int8')
            self.add_random()
            self.add_random()
        self._n = n
        self._field_width = 5
        self._board_size = self._field_width * n + n + 1

    def pretty_str(self):
        result = '-' * self._board_size + '\n'
        for row in self.state:
            result += '|'
            for value in row:
                result += (f'{2**value:^{self._field_width}d}' if value > 0 else ' ' * self._field_width) + '|'
            result += '\n'
            result += '-' * self._board_size + '\n'
        return result

    def __str__(self):
        return str(self.state)

    def add_random(self):
        empty_x, empty_y = np.where(self.state == 0)
        if len(empty_x) == 0:
            return
        index = random.randint(0, len(empty_x) - 1)
        self.state[empty_x[index], empty_y[index]] = 1 if np.random.choice([True, False], p=[0.9, 0.1]) else 2

    def is_board_full(self):
        return np.count_nonzero(self.state) == self._n * self._n

    def make_move(self, direction: int):
        self.state = np.rot90(self.state, direction)
        result = self._left()
        self.state = np.rot90(self.state, -direction)
        return result

    def dry_move(self, direction: int):
        state_backup = self.state.copy()
        self.state = np.rot90(self.state, direction)
        result = self._left()
        self.state = np.rot90(self.state, -direction)
        self.state = state_backup
        return result

    def _left(self):
        merged_values = []
        move_counter = 0
        for row in range(self._n):
            i, j = 0, 1
            while i < self._n and j < self._n:
                if self.state[row][j] == 0:
                    j += 1
                    continue
                if self.state[row][i] == self.state[row][j]:
                    move_counter += 1
                    self.state[row][i] += 1
                    self.state[row][j] = 0
                    merged_values.append(self.state[row][i])
                    i += 1
                    j += 1
                elif self.state[row][i] == 0:
                    move_counter += 1
                    self.state[row][i] = self.state[row][j]
                    self.state[row][j] = 0
                    j += 1
                else:
                    i += 1
                    if i == j:
                        j += 1
        return BoardMoveResult(move_count=move_counter, merged_values=merged_values)

    def _left_check(self):
        for row in range(self._n):
            i, j = 0, 1
            while i < self._n and j < self._n:
                if self.state[row][j] == 0:
                    j += 1
                    continue
                if self.state[row][i] == self.state[row][j]:
                    return True
                elif self.state[row][i] == 0:
                    return True
                else:
                    i += 1
                    if i == j:
                        j += 1
        return False

    def available_moves(self):
        moves = []
        for direction in range(4):
            self.state = np.rot90(self.state, direction)
            if self._left_check():
                moves.append(direction)
            self.state = np.rot90(self.state, -direction)
        return moves

    def can_move(self):
        return len(self.available_moves()) != 0

    def max_number(self):
        return 2 ** np.amax(self.state)

    def sum(self):
        return sum([2 ** i if i > 0 else 0 for i in self.state.flatten()])
