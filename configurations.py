from helper_functions import *
import tensorflow as tf
from recordclass import recordclass

import os

DISCOUNT_FACTOR = 0.95
LR = 0.0001
SAVE_MODELS_FREQ = 50_000
LOG_PROGRESS_FREQ = 1_500
CONF_TO_TEST_NUMBER = 10000
USE_GPU = False

DEEP_LAYERS_SIZES = [[16, 14, 12, 10, 8, 6, 4], [16] * 10, [8] * 10, [256], [32] * 5, [64] * 3, [256] * 3, [4] * 10]

OUTPUT_ACTIVATIONS = ['linear']
OPTIMIZERS = ['adam', 'sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
LOSSES = ['mse', 'mae']
STATE_MAP_FUNCTIONS = [div_by_max, do_nothing, norm_16, game_power_2, real_power_2]
BATCH_SIZES = [8, 32, 64, 1000]
DOUBLE_Q_LEARNING_OPTS = [True, False]
MEMORY_SIZES = [3_000, 10_000, 100_000, 1_000_000]
LEARN_EACH_OPTS = [4, 10, 30, 100]
MIN_EPS_OPTS = [0, 0.07, 0.1, 0.15]
REWARD_FUNCS = [count_reward, power_reward, punishing_power_reward, dynamic_punishing_power_reward, value_reward,
                time_reward_minus, time_reward_plus, dynamic_time_reward_plus, time_reward_plus_minus,
                dynamic_time_reward_plus_minus]
ALLOW_ILLEGAL_OPTS = [True, False]
EQUAL_DONES_OPTS = [True, False]
EQUAL_DIRECTIONS_OPTS = [True, False]
DRY_OPTS = [True, False]
CRUCIAL_OPTS = [True, False]

STEPS = 750_000
ZERO_EPS_STEP = STEPS - 100_000
MEMORY_STATS_EACH_STEPS = 100
DIAG_EVALUATE_EACH_GAMES = 100
EPSILON_CONSTATNS = [30_00, 50_000, 100_000]
EVALUATION_STEPS = 25_000
UPDATE_TARGETS_EACH_TRAIN_OPTS = [4, 10, 100]
CONVOLUTIONAL_LAYERS = [[]]
CONV_ACTIVATIONS = ['linear', 'relu']

####
# Normalized:
####

REWARD_FUNCS = [punishing_normalized_count_reward, dynamic_normalized_count_reward, normalized_count_reward,
                dynamic_punishing_normalized_count_reward]
STATE_MAP_FUNCTIONS = [norm_16, div_by_max]

######
# Big net
######
DEEP_LAYERS_SIZES = [[500, 300, 200, 200, 100]]
CONVOLUTIONAL_LAYERS = [[]]

#####
# Convolutions
#####
DEEP_LAYERS_SIZES = [[256], [256, 256], [64] * 3, [128], [128 * 2]]
CONVOLUTIONAL_LAYERS = [[(128, 3), (128, 2)], [(256, 3), (256, 2)], [(64, 3), (64, 2)], [(128, 3), (256, 2)]]

#####
# Fast testnig
#####
SAVE_MODELS_FREQ = 50_000
LOG_PROGRESS_FREQ = 1_500
STEPS = 150_000
ZERO_EPS_STEP = STEPS - 25_000
MEMORY_STATS_EACH_STEPS = 1_000
DIAG_EVALUATE_EACH_GAMES = 50
EPSILON_CONSTATNS = [3_000, 5_000, 10_000]

#####
# Long training
#####
SAVE_MODELS_FREQ = 50_000
LOG_PROGRESS_FREQ = 1_500
STEPS = 750_000
ZERO_EPS_STEP = STEPS - 100_000
MEMORY_STATS_EACH_STEPS = 50_000
DIAG_EVALUATE_EACH_GAMES = 1000
EPSILON_CONSTATNS = [30_00, 50_000, 100_000]

#####
# Really long training
#####

SAVE_MODELS_FREQ = 250_000
LOG_PROGRESS_FREQ = 2_500
STEPS = 10_000_000
ZERO_EPS_STEP = STEPS - 500_000
MEMORY_STATS_EACH_STEPS = 100_000
DIAG_EVALUATE_EACH_GAMES = 10_000
EPSILON_CONSTATNS = [30_00, 50_000, 100_000]

TrainingConf = recordclass('TrainingConf',
                           'allow_illegal min_eps optimizer loss conv_activation conv_layers layers_size output_activation batch_size '
                           'update_targets_each learn_each memory_size state_map_function double_q reward_func '
                           'epsilon_constant equal_dones equal_directions crucial dry')

ALL_OPTIONS = [ALLOW_ILLEGAL_OPTS, MIN_EPS_OPTS, OPTIMIZERS, LOSSES, CONV_ACTIVATIONS, CONVOLUTIONAL_LAYERS,
               DEEP_LAYERS_SIZES, OUTPUT_ACTIVATIONS, BATCH_SIZES, UPDATE_TARGETS_EACH_TRAIN_OPTS, LEARN_EACH_OPTS,
               MEMORY_SIZES, STATE_MAP_FUNCTIONS, DOUBLE_Q_LEARNING_OPTS, REWARD_FUNCS, EPSILON_CONSTATNS,
               EQUAL_DONES_OPTS, EQUAL_DIRECTIONS_OPTS, CRUCIAL_OPTS, DRY_OPTS]

TOTAL_RANDOM = TrainingConf(False, 1, 'adam', 'mae', None, [], [8] * 10, 'linear', 64, 1000000000000, 100000000000,
                            10000, do_nothing, False, power_reward, 100000000, False, False, False, False)

BIG_NET_NORMALIZED_DPNCR = TrainingConf(True, 0.1, 'adam', 'mae', None, [], [500, 300, 200, 200, 100], 'linear', 64,
                                        100,
                                        100, 100_000, norm_16, False, dynamic_punishing_normalized_count_reward, 50_000,
                                        False, False, False, False)

BIG_NET_NORMALIZED_NCR = TrainingConf(False, 0.1, 'adam', 'mae', None, [], [500, 300, 200, 200, 100], 'linear', 64, 100,
                                      100, 100_000, norm_16, False, normalized_count_reward, 50_000,
                                      False, False, False, False)

BIG_NET_NORMALIZED_NCR_SMART_MEM = TrainingConf(False, 0.1, 'adam', 'mae', None, [], [500, 300, 200, 200, 100],
                                                'linear', 64, 100,
                                                100, 100_000, norm_16, False, normalized_count_reward, 50_000,
                                                True, True, True, False)

BIG_NET_NORMALIZED_DPNCR_DRY = TrainingConf(False, 0.1, 'adam', 'mae', None, [], [500, 300, 200, 200, 100], 'linear',
                                            64, 100,
                                            100, 100_000, norm_16, False, dynamic_punishing_normalized_count_reward,
                                            50_000,
                                            False, False, False, True)

CONV_DPNCR = TrainingConf(True, 0.1, 'adam', 'mae', 'linear', [(128, 3), (128, 2)], [256], 'linear', 64, 100,
                          100, 100_000, norm_16, False, dynamic_punishing_normalized_count_reward, 50_000,
                          False, False, False, False)

CONV_NCR = TrainingConf(False, 0.1, 'adam', 'mae', 'linear', [(128, 3), (128, 2)], [256], 'linear', 64, 100,
                        100, 100_000, norm_16, False, normalized_count_reward, 50_000,
                        False, False, False, False)

CONV_DPNCR_SMART_MEM = TrainingConf(True, 0.1, 'adam', 'mae', 'linear', [(128, 3), (128, 2)], [256], 'linear', 64, 100,
                                    100, 100_000, norm_16, False, dynamic_punishing_normalized_count_reward, 50_000,
                                    True, True, True, False)

CONV_DPNCR_DRY = TrainingConf(False, 0.1, 'adam', 'mae', 'linear', [(128, 3), (128, 2)], [256], 'linear', 64, 100,
                              100, 100_000, norm_16, False, dynamic_punishing_normalized_count_reward, 50_000,
                              False, False, False, True)

USE_SPECIFIC_CONF = False
SPECIFIC_CONFIGURATIONS = [BIG_NET_NORMALIZED_DPNCR, BIG_NET_NORMALIZED_NCR, CONV_DPNCR, CONV_NCR]

STATE_MAP_FUNCTIONS_DICT = {'div_by_max': div_by_max, 'do_nothing': do_nothing, 'normalized_16': norm_16,
                            'gp2': game_power_2, 'rp2': real_power_2}

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
