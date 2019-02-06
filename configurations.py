from helper_functions import *
import tensorflow as tf
from recordclass import recordclass

import os

DISCOUNT_FACTOR = 0.95
SAVE_MODELS_FREQ = 5_000
LOG_PROGRESS_FREQ = 1_500
CONF_TO_TEST_NUMBER = 10000
USE_GPU = False

LAYERS_SIZES = [[16, 14, 12, 10, 8, 6, 4], [16] * 10, [8] * 10, [256], [32] * 5, [64] * 3, [256] * 3, [4] * 10]
OUTPUT_ACTIVATIONS = ['linear']
OPTIMIZERS = ['adam', 'sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
LOSSES = ['mse', 'mae', tf.losses.huber_loss]
STATE_MAP_FUNCTIONS = [div_by_max, do_nothing, norm_16, game_power_2, real_power_2]
BATCH_SIZES = [8, 32, 64, 1000]
DOUBLE_Q_LEARNING_OPTS = [True, False]
MEMORY_SIZES = [3_000, 10_000, 100_000, 1_000_000]
LEARN_EACH_OPTS = [4, 10, 30, 100]
MIN_EPS_OPTS = [0, 0.07, 0.1, 0.15, 0.2]
REWARD_FUNCS = [count_reward, power_reward, punishing_power_reward, dynamic_punishing_power_reward, value_reward,
                time_reward_minus, time_reward_plus, dynamic_time_reward_plus, time_reward_plus_minus,
                dynamic_time_reward_plus_minus]
ALLOW_ILLEGAL_OPTS = [True, False]
EQUAL_DONES_OPTS = [True, False]
EQUAL_DIRECTIONS_OPTS = [True, False]
DRY_OPTS = [True, False]
CRUCIAL_OPTS = [True, False]

# Diagnostic testing

STEPS = 100_000
ZERO_EPS_STEP = STEPS - 20_000
MEMORY_STATS_EACH_STEPS = 100
DIAG_EVALUATE_EACH_GAMES = 5
EPSILON_CONSTATNS = [3_000, 5_000, 10_000, 30_000]
EVALUATION_STEPS = 25_000
UPDATE_TARGETS_EACH_TRAIN_OPTS = [4, 10, 100]

TrainingConf = recordclass('TrainingConf',
                           'allow_illegal min_eps optimizer loss layers_size output_activation batch_size '
                           'update_targets_each learn_each memory_size state_map_function double_q reward_func '
                           'epsilon_constant equal_dones equal_directions crucial dry')

ALL_OPTIONS = [ALLOW_ILLEGAL_OPTS, MIN_EPS_OPTS, OPTIMIZERS, LOSSES, LAYERS_SIZES, OUTPUT_ACTIVATIONS,
               BATCH_SIZES, UPDATE_TARGETS_EACH_TRAIN_OPTS, LEARN_EACH_OPTS, MEMORY_SIZES, STATE_MAP_FUNCTIONS,
               DOUBLE_Q_LEARNING_OPTS, REWARD_FUNCS, EPSILON_CONSTATNS, EQUAL_DONES_OPTS, EQUAL_DIRECTIONS_OPTS,
               CRUCIAL_OPTS, DRY_OPTS]

TOTAL_RANDOM = TrainingConf(False, 1, 'adam', 'mae', [8] * 10, 'linear', 64, 1000000000000, 100000000000, 10000,
                            do_nothing, False, power_reward, 100000000, False, False, False, False)

USE_SPECIFIC_CONF = False
SPECIFIC_CONFIGURATIONS = [TOTAL_RANDOM]

STATE_MAP_FUNCTIONS_DICT = {'div_by_max': div_by_max, 'do_nothing': do_nothing, 'normalized_16': norm_16,
                            'gp2': game_power_2, 'rp2': real_power_2}

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
