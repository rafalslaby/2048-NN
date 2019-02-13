import csv
import datetime
import sys
import os
from pathlib import Path
from paths_helpers import *
from neural_network import make_model, format_for_input, is_model_one_hot, one_hot_encode_input
from helper_functions import *
from configurations import *
from q_agent import DQNAgent
from keras.models import load_model
import env_2048
from replay_memories import *
from recordclass import recordclass
from itertools import count

score_csv_header = ['step', 'game', 'eps', 'max_tile', 'sum', 'total_reward']
suffixes = ['step', 'tile', 'sum', 'rew']
prefixes = ['max', 'mean', 'max_100', 'mean_100']
progress_csv_header = ['time', 'step', 'game', 'eps', 'max_step', 'max_tile', 'max_sum', 'max_rew', 'mean_step',
                       'mean_tile', 'mean_sum', 'mean_rew', 'max_100_step', 'max_100_tile', 'max_100_sum',
                       'max_100_rew', 'mean_100_step', 'mean_100_tile', 'mean_100_sum', 'mean_100_rew']

GameResult = namedtuple('GameResult', 'steps max_tile sum total_reward')
datetime_format = '%Y-%m-%d_%H:%M:%S.%f'

TrainingConf = recordclass('TrainingConf',
                           'allow_illegal min_eps optimizer loss conv_activation conv_layers layers_size output_activation batch_size '
                           'update_targets_each learn_each memory_size state_map_function double_q reward_func '
                           'epsilon_constant equal_dones equal_directions crucial dry')


# TODO: try convolutions, try really big network, try different deep activation


def remove_contradictions_from_config(config: TrainingConf):
    if config.dry:
        config.allow_illegal = False


def configuration_gen(root_dir):
    datetime_stamp_format = '%Y_%m_%d_%H_%M'
    while True:
        if USE_SPECIFIC_CONF:
            c = random.choice(SPECIFIC_CONFIGURATIONS)
            conf_dir = Path(root_dir) / get_configuration_one_dir(c) / pathlib.Path(
                datetime.datetime.now().strftime(datetime_stamp_format))
        else:
            c = TrainingConf(*[random.choice(opt) for opt in ALL_OPTIONS])
            remove_contradictions_from_config(c)
            conf_dir = Path(root_dir) / get_configuration_one_dir(c) / pathlib.Path(
                datetime.datetime.now().strftime(datetime_stamp_format))
        conf_dir.mkdir(parents=True, exist_ok=True)

        model_file = conf_dir / 'model.h5'
        model = make_model(c.layers_size, c.optimizer, c.output_activation, loss=c.loss, conv_layers=c.conv_layers,
                           conv_activation=c.conv_activation)

        if c.double_q:
            target_model_file = conf_dir / 'target_model.h5'
            target_model = make_model(c.layers_size, c.optimizer, c.output_activation, loss=c.loss,
                                      conv_layers=c.conv_layers, conv_activation=c.conv_activation)
        else:
            target_model = model
            target_model_file = model_file
        yield c, model, target_model, model_file, target_model_file, conf_dir


def training_result_row(step, game, eps, results):
    mean_results = np.mean(results, axis=0)
    mean_last_100 = np.mean(results[-100:], axis=0)
    max_results = np.max(results, axis=0)
    max_results_last_100 = np.max(results[-100:], axis=0)
    return [datetime.datetime.now().strftime(datetime_format), step, game, eps, *max_results, *mean_results,
            *max_results_last_100, *mean_last_100]


def save_results(file_path, step, game, eps, results):
    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(progress_csv_header)
        csv_writer.writerow(training_result_row(step, game, eps, results))


def extract_state_map_func(path: str):
    return STATE_MAP_FUNCTIONS_DICT[next(filter(lambda f: path.find(f) != -1, STATE_MAP_FUNCTIONS_DICT.keys()), 'do_nothing')]


def examine_model(conf_dir, games, out_file=sys.stdout, verbose=False, model=None,
                  extra_prefs_file=open(os.devnull, 'w'), render=False, render_fps=10, game_over_sleep=1000):
    state_map_func = extract_state_map_func(str(conf_dir))
    # allow_illegal = re.search(r'ill_([^\\]+)', str(conf_dir)).group(1) == 'True'
    directions = ['left', 'up', 'right', 'down']
    model = model or load_model(str(pathlib.Path(conf_dir) / 'model.h5'))
    input_format_func = one_hot_encode_input if is_model_one_hot(model) else format_for_input
    env = env_2048.Env2048(gui_visualization=render, render_fps=render_fps, game_over_sleep=game_over_sleep)
    for game in range(games):
        moves = [0] * 4
        illegal_tries = 0
        pref_counters = [0] * 4
        directions_scores = np.zeros(4)
        done = False
        while not done:
            from_state = env.state()
            q_table = model.predict(
                input_format_func([state_map_func(from_state)]), batch_size=1)[0]
            choices = env.act_space()

            move = choose_best_valid(q_table, choices)
            directions_preference = np.argsort(-q_table)
            pref_counters[directions_preference[0]] += 1
            directions_scores += directions_preference[::-1]
            illegal_tries += np.argwhere(directions_preference == move)[0][0]
            moves[move] += 1
            step_result = env.step(move)
            done = step_result.is_game_over
            if verbose:
                print(from_state, file=out_file)
                print(" LEFT        UP        RIGHT      DOWN", file=out_file)
                print(q_table, file=out_file)
                print(directions[move], "from ", [directions[choice] for choice in choices], '\n', file=out_file)
        if verbose:
            print(env.state(), file=out_file)
            all_moves = sum(moves)
            print(f"GAME OVER! Score: {env.score()}; steps: {all_moves}", file=out_file)
            print('Directions:', [n / all_moves for n in moves], file=out_file)
            print('Illegal tries', illegal_tries / all_moves, illegal_tries, 'in', all_moves, 'steps', file=out_file)
            print('Pref counters', [n / all_moves for n in pref_counters], file=out_file)
            print('Direction scores', directions_scores / (all_moves * 4), file=out_file)
            print('Pref counters', [n / all_moves for n in pref_counters], file=extra_prefs_file)
            print('\n\n\n', file=out_file)

        env.reset()


def train_configuration(steps, c, model, target_model, model_file, target_model_file, conf_dir, evaluate=False,
                        extra_prefs_file=open(os.devnull, 'w'), render=False, render_fps=10, game_over_sleep=1000):
    name_prefix = 'evaluate_' if evaluate else 'train_'
    with open(conf_dir / f'{name_prefix}scores.csv', 'w', newline='') as scores_file, \
            open(conf_dir / f'{name_prefix}progress.csv', 'w', newline='') as progress_file, \
            open(conf_dir / 'memory_stats.txt', 'a') as mem_stats_file:
        epsilon_func = (lambda x: 0) if evaluate else make_decreasing_epsilon(c.epsilon_constant, c.min_eps)
        if c.equal_directions or c.equal_dones or c.crucial:
            memory_keeper = SmartReplayMemory(c.memory_size, c.equal_directions, c.equal_dones, c.crucial)
        else:
            memory_keeper = ReplayMemory(c.memory_size)

        agent = DQNAgent(0, model, target_model, epsilon_func, memory_keeper, c.batch_size, c.learn_each,
                         DISCOUNT_FACTOR, c.update_targets_each, choose_best_valid, conf_dir / 'diag')

        if evaluate:
            agent.epsilon_func = lambda x: 0
        (conf_dir / 'diag').mkdir(parents=True, exist_ok=True)

        results = []
        env = env_2048.Env2048(gui_visualization=render, render_fps=render_fps, game_over_sleep=game_over_sleep)
        games_counter = 0
        scores_writer = csv.writer(scores_file)
        scores_writer.writerow(score_csv_header)
        progress_writer = csv.writer(progress_file)
        progress_writer.writerow(progress_csv_header)

        total_reward = 0
        last_step_end = 0
        for step in range(1, steps + 1):
            if evaluate:
                step_result, reward, _ = agent.one_full_step(env.act_space(), c, env, evaluate)
            elif c.dry:
                step_result, reward, _ = agent.test_all_do_best(c, env, evaluate)
            elif c.allow_illegal:
                step_result, reward = agent.play_until_made_move(c, env, evaluate)
            else:
                step_result, reward, _ = agent.one_full_step(env.act_space(), c, env, evaluate)

            total_reward += reward
            done = step_result.is_game_over
            if done:
                games_counter += 1
                score = env.score()
                results.append(GameResult(step - last_step_end, score.max_tile, score.sum, total_reward))
                last_step_end = step
                scores_writer.writerow([step, games_counter, agent.epsilon, score.max_tile, score.sum, total_reward])
                total_reward = 0
                env.reset()
                if not evaluate and games_counter % DIAG_EVALUATE_EACH_GAMES == 0:
                    with open(conf_dir / 'diag' / f'eval_at_{games_counter}_game.txt', 'w') as eval_file:
                        print(f"{step} {games_counter} {agent.epsilon}", file=eval_file)
                        examine_model(conf_dir, 1, eval_file, True, model, extra_prefs_file=extra_prefs_file)

            if not evaluate:
                if step % SAVE_MODELS_FREQ == 0:
                    model.save(model_file)
                    if model is not target_model:
                        target_model.save(target_model_file)
                    scores_file.flush()
                    progress_file.flush()
                    mem_stats_file.flush()

                if step > ZERO_EPS_STEP:
                    agent.epsilon_func = lambda x: 0

                if step % MEMORY_STATS_EACH_STEPS == 0:
                    print(step, games_counter, agent.epsilon, *agent.percentage_memory_stats(), file=mem_stats_file)

            if step % LOG_PROGRESS_FREQ == 0 and len(results) > 1:
                progress_writer.writerow(training_result_row(step, games_counter, agent.epsilon, results))
                progress_file.flush()

        done_file = conf_dir / f'{name_prefix}done.csv'
        save_results(done_file, last_step_end, games_counter, agent.epsilon, results)
        model.save(model_file)
        if model is not target_model:
            target_model.save(target_model_file)
        print(training_result_row(last_step_end, games_counter, agent.epsilon, results))


def start_training(out_dir='results', render=False, render_fps=1, game_over_sleep=1000):
    configuration_counter = count()
    for c, model, target_model, model_file, target_model_file, conf_dir in configuration_gen(out_dir):
        print(next(configuration_counter), ':', str(conf_dir))
        if len(list(conf_dir.parent.rglob('**/evaluate_done.csv'))) != 0:
            print('Configuration already done. If you want to train it again rename evaluate_done.csv')
            continue
        with open(conf_dir / 'extra_pref.txt', 'w') as extra_prefs_file:
            train_configuration(STEPS, c, model, target_model, model_file, target_model_file, conf_dir, False,
                                extra_prefs_file, render=render, render_fps=render_fps, game_over_sleep=game_over_sleep)

        train_configuration(EVALUATION_STEPS, c, model, target_model, model_file, target_model_file, conf_dir,
                            evaluate=True, render=render, render_fps=render_fps, game_over_sleep=game_over_sleep)


def profile_training(out_dir='results'):
    import pstats
    import cProfile
    c, model, target_model, model_file, target_model_file, conf_dir = next(configuration_gen(out_dir))
    with open(conf_dir / 'train_cum.txt', 'w') as tc, open(conf_dir / 'train_time.txt', 'w') as tt, open(
            conf_dir / 'eval_cum.txt', 'w') as ec, open(conf_dir / 'eval_time.txt', 'w') as et:
        cProfile.run('train(STEPS, c, model, target_model, model_file, target_model_file, conf_dir, False)',
                     'stats')
        pstats.Stats('stats', stream=tc).sort_stats('cumulative').print_stats(100)
        pstats.Stats('stats', stream=tt).sort_stats('time').print_stats(100)

        cProfile.run(
            'train(EVALUATION_STEPS, c, model, target_model, model_file, target_model_file, conf_dir, True)',
            'stats')
        p = pstats.Stats('stats', stream=ec).sort_stats('cumulative')
        p.print_stats(100)
        p = pstats.Stats('stats', stream=et).sort_stats('time')
        p.print_stats(100)


def start_training_multiprocess(out_dir='results', render=False, render_fps=1, game_over_sleep=1000, jobs=1):
    from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(start_training, out_dir, render, render_fps, game_over_sleep) for _ in range(jobs)]
        wait(futures, return_when=ALL_COMPLETED)


if __name__ == '__main__':
    start_training()
