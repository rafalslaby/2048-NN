import numpy as np
from collections import namedtuple

# just for typing
StepResult = namedtuple("StepResult", "state board_move_result is_game_over")


def choose_best_valid(q_table, choices):
    i_max = choices[0]
    for i in range(len(q_table)):
        if q_table[i] > q_table[i_max] and i in choices:
            i_max = i
    return i_max


def numpy_map_1d(func, arr, result_dtype=None):
    return np.fromiter((func(val) for val in arr), result_dtype or arr.dtype)


def greedy_strategy(state, *args, **kwargs):
    return np.argmax(state)


def make_constant_epsilon(value):
    return lambda run_number: value


def do_nothing(value):
    return value


bases = np.ones(shape=(4, 4), dtype=np.int8) * 2


def real_power_2(value):
    return np.power(bases, value, dtype='int32')


real_power_2.short_name = 'rp2'


def _power_2_or_zero(value):
    if value == 0:
        return 0
    return 2 ** value


def game_power_2(value):
    return numpy_map_1d(_power_2_or_zero, value.flatten(), 'int32').reshape((4, 4))


game_power_2.short_name = 'gp2'


def norm_16(state):
    return state / 16


def div_by_max(state):
    return state / np.amax(state)


def make_decreasing_epsilon(constant, min_eps):
    return lambda run_number: max(min_eps, constant / (constant + run_number))


def count_reward(result: StepResult):
    return len(result.board_move_result)


count_reward.short_name = 'cr'


def normalized_count_reward(result: StepResult):
    return len(result.board_move_result) / 8


count_reward.short_name = 'ncr'


def dynamic_normalized_count_reward(result: StepResult):
    return len(result.board_move_result - 1) / 8


count_reward.short_name = 'ncr'


def punishing_normalized_count_reward(result: StepResult):
    if result.is_game_over:
        return -1
    return (len(result.board_move_result) - 1) / 8


count_reward.short_name = 'pncr'


def dynamic_punishing_normalized_count_reward(result: StepResult):
    if result.is_game_over:
        return -1
    return (len(result.board_move_result)) / 8


count_reward.short_name = 'dpncr'


def power_reward(result: StepResult):
    return sum(2 ** value for value in result.board_move_result.merged_values)


power_reward.short_name = 'pr'


def punishing_power_reward(result: StepResult):
    return power_reward(result) - 100 * result.is_game_over


punishing_power_reward.short_name = 'ppr'


def dynamic_punishing_power_reward(result: StepResult):
    reward = power_reward(result)
    reward = reward if reward > 0 else -10
    return reward - 100 * result.is_game_over


dynamic_punishing_power_reward.short_name = 'dppr'


def value_reward(result: StepResult):
    return sum(result.board_move_result.merged_values)


value_reward.short_name = 'vr'


def time_reward_minus(result: StepResult):
    if result.is_game_over:
        return -1
    return 0


time_reward_minus.short_name = 'trm'


def time_reward_plus(result: StepResult):
    if result.is_game_over:
        return 0
    return 1


time_reward_plus.short_name = 'trp'


def dynamic_time_reward_plus(result: StepResult):
    if result.is_game_over or len(result.board_move_result.merged_values) == 0:
        return 0
    return 1


dynamic_time_reward_plus.short_name = 'dtrp'


def time_reward_plus_minus(result: StepResult):
    if result.is_game_over:
        return -10
    return 1


time_reward_plus_minus.short_name = 'trpm'


def dynamic_time_reward_plus_minus(result: StepResult):
    if result.is_game_over:
        return -10
    if len(result.board_move_result.merged_values) == 0:
        return -1
    return 1


dynamic_time_reward_plus_minus.short_name = 'dtrpm'
