from helper_functions import *
import tensorflow as tf
from recordclass import recordclass

import os

ONE_HOT = False
DISCOUNT_FACTOR = 0.95
LR = 0.0001
SAVE_MODELS_FREQ = 50_000
LOG_PROGRESS_FREQ = 1_500
CONF_TO_TEST_NUMBER = 10000
USE_GPU = False

DEEP_LAYERS_SIZES = [[16, 14, 12, 10, 8, 6, 4], [16] * 10, [8] * 10, [256], [32] * 5, [64] * 3, [256] * 3, [4] * 10, [500, 300, 200, 200, 100],
    [4]*6,[8]*6]

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
                dynamic_time_reward_plus_minus, punishing_normalized_count_reward, dynamic_normalized_count_reward, normalized_count_reward,
                dynamic_punishing_normalized_count_reward]
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
CONVOLUTIONAL_LAYERS = [[(128, 3), (128, 2)], [(256, 3), (256, 2)], [(64, 3), (64, 2)], [(128, 3), (256, 2)],[],[],[],[],[],[],[]]
CONV_ACTIVATIONS = ['linear', 'relu']

### Has to be turned manually because it's not compatible with every state map and reward function
ONE_HOT = True
STATE_MAP_FUNCTIONS = [do_nothing]
REWARD_FUNCS = [normalized_count_reward]


####
# Only normalized:
####

REWARD_FUNCS = [punishing_normalized_count_reward, dynamic_normalized_count_reward, normalized_count_reward,
                dynamic_punishing_normalized_count_reward]
STATE_MAP_FUNCTIONS = [norm_16, div_by_max]

# #####
# Big net
# #####
DEEP_LAYERS_SIZES = [[256]*5, [256]*6, [256]*7, [900,500,300,200,100],[64]*7,[8]*10]
CONVOLUTIONAL_LAYERS = [[]]

#####
# Convolutions
#####
DEEP_LAYERS_SIZES = [[256], [256, 256], [64] * 3, [128], [128 * 2]]
CONVOLUTIONAL_LAYERS = [[(128, 3), (128, 2)], [(256, 3), (256, 2)], [(64, 3), (64, 2)], [(128, 3), (256, 2)]]


### Favor best configurations

LOSSES = ['mse', 'mae', 'mae', 'mae']
REWARD_FUNCS = [punishing_normalized_count_reward, dynamic_normalized_count_reward, normalized_count_reward,
                dynamic_punishing_normalized_count_reward,normalized_count_reward,normalized_count_reward,normalized_count_reward]

#####
# Fast testnig
#####
SAVE_MODELS_FREQ = 10_000
LOG_PROGRESS_FREQ = 1_500
STEPS = 300_000
ZERO_EPS_STEP = STEPS - 80_000
MEMORY_STATS_EACH_STEPS = 1_000
DIAG_EVALUATE_EACH_GAMES = 50
EPSILON_CONSTATNS = [3_000, 5_000, 10_000]

#####
# Long training
#####
SAVE_MODELS_FREQ = 25_000
LOG_PROGRESS_FREQ = 1_500
STEPS = 750_000
ZERO_EPS_STEP = STEPS - 100_000
MEMORY_STATS_EACH_STEPS = 50_000
DIAG_EVALUATE_EACH_GAMES = 50
EPSILON_CONSTATNS = [10_00, 25_000, 50_000]

#####
# Really long training
#####

SAVE_MODELS_FREQ = 250_000
LOG_PROGRESS_FREQ = 2_500
STEPS = 5_000_000
ZERO_EPS_STEP = STEPS - 2_000_000
MEMORY_STATS_EACH_STEPS = 100_000
DIAG_EVALUATE_EACH_GAMES = 1_000
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

#ill1_em000_Adamax_mae_linear__64_64_64_64_64_64_64_linear_batch_1000_tUpdF4_learnF100_do_nothing_ddq_0_ncr_epsC3000_RMD1000000_dry0/2019_02_12_00_29/
#ill0_em015_Adadelta_mae_linear__256_256_256_256_256_linear_batch_32_tUpdF100_learnF4_div_by_max_ddq_1_ncr_epsC50000_SRM1000000_dry1/2019_02_10_04_37

BEST_ONE_HOT = TrainingConf(True,0,'Adamax','mae','linear',[],[64]*7,'linear',1000,4,100,1000000,do_nothing,False, normalized_count_reward,10_000,True, False, False, False)
BEST_BIG_NET = TrainingConf(False,0.15,'Adadelta','mae','linear',[],[256]*5,'linear',32,100,4,1000000,div_by_max,True, normalized_count_reward,50_000,True, False, False, False)

USE_SPECIFIC_CONF = False
SPECIFIC_CONFIGURATIONS = [BEST_ONE_HOT, BEST_BIG_NET]

STATE_MAP_FUNCTIONS_DICT = {'div_by_max': div_by_max, 'do_nothing': do_nothing, 'norm_16': norm_16,
                            'gp2': game_power_2, 'rp2': real_power_2}

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
