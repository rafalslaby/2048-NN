import functools
import random
import numpy as np
from collections import namedtuple, deque
from board import Board


class SmartReplayMemory:
    def __init__(self, maxlen: int, equal_directions=False, equal_dones=False, crucial=False):
        """
        Supports better experience sampling
        :param maxlen: max memory length, when max is reached the oldest memory will be discarded
        :param equal_directions: sample experiences equal experience number in each direction
        :param equal_dones: sample sample extra max(n//10,1) moves that resulted in game end
        :param crucial: sample extra max(n//10,1) moves when there was 2 or less choices
        """

        self.moves_in_each_direction = [deque(maxlen=maxlen // 4) for _ in range(4)]
        self.done_moves = deque(maxlen=maxlen // 10)
        self.equal_directions = equal_directions
        self.mem_all = deque(maxlen=maxlen)
        self.equal_dones = equal_dones
        self.crucial = crucial
        if crucial:
            self.crucial_moves = deque(maxlen=maxlen // 10)

    def append(self, exp):
        if exp.done:
            self.done_moves.append(exp)
        if self.crucial and self.move_is_crucial(exp):
            self.crucial_moves.append(exp)
        if self.equal_directions:
            self.moves_in_each_direction[exp.action].append(exp)
        else:
            self.mem_all.append(exp)

    def move_is_crucial(self, exp):
        return len(Board(4, exp.from_state).available_moves()) < 3

    def __len__(self):
        if self.equal_directions:
            return sum([len(mem) for mem in self.moves_in_each_direction])
        else:
            return len(self.mem_all)

    def sample(self, n):
        """
        WARNING - will return more than n if extra flags were specified in ctor
        """
        if self.equal_directions:
            result = functools.reduce(lambda x, y: x + y,
                                      (random.sample(mem, n // 4) for mem in self.moves_in_each_direction if
                                       len(mem) >= n // 4), [])
        else:
            result = random.sample(self.mem_all, n)

        if self.equal_dones:
            done_sample_len = max(1, n // 10)
            if len(self.done_moves) >= done_sample_len:
                result += random.sample(self.done_moves, done_sample_len)
        if self.crucial:
            crucial_sample_len = max(1, n // 4)
            if len(self.crucial_moves) >= crucial_sample_len:
                result += random.sample(self.crucial_moves, crucial_sample_len)
        return result

    def memory_stats(self):
        if self.equal_directions:
            return self.__len__(), len(self.done_moves), [len(mem) for mem in self.moves_in_each_direction]
        else:
            self._np_memory = np.array(self.mem_all)
            done_count = self._done_experiences()
            return self._np_memory.shape[0], done_count, self._directions_histogram()

    def percentage_memory_stats(self):
        stats = self.memory_stats()
        return stats[0], stats[1] / stats[0], stats[2] / np.sum(stats[2]), (
            len(self.crucial_moves) / stats[0] if self.crucial else None)

    def __str__(self):
        """
        SRMDC = Smart Replay Memory Dynamic Crucial
        """
        return ('S' if self.equal_directions else '') + 'RM' + ('D' if self.equal_dones else '') + (
            'C' if self.crucial else '')

    def _directions_histogram(self):
        return np.unique(self._np_memory[:, 1], return_counts=True)[1]

    def _done_experiences(self):
        return np.unique(self._np_memory[:, 4], return_counts=True)[1]
