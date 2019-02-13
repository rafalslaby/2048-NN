import os, csv, datetime, sys
import pathlib
import matplotlib.pyplot as plt
from helper_functions import *

plt.rcParams.update({'font.size': 13})

ProgressRecord = namedtuple('ProgressRecord',
                            ['time', 'step', 'game', 'eps', 'max_step', 'max_tile', 'max_sum', 'max_rew', 'mean_step',
                             'mean_tile', 'mean_sum', 'mean_rew', 'max_100_step', 'max_100_tile', 'max_100_sum',
                             'max_100_rew', 'mean_100_step', 'mean_100_tile', 'mean_100_sum', 'mean_100_rew'])

ScoresRecord = namedtuple('ScoresRecord', ['step', 'game', 'eps', 'max_tile', 'sum', 'total_reward'])


def iter_dirs_with_file(filename, specific_dir=None):
    path = pathlib.Path(specific_dir) if specific_dir else pathlib.Path('.')
    for file in path.glob(f"**/{filename}"):
        if file.stat().st_size > 0:
            yield file.parent


def extract_csv_rows(path, parsing_func):
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, 0)  # skip header
        return [parsing_func(row) for row in reader]


def parse_progress_stats_row(stats_row):
    date = datetime.datetime.strptime(stats_row[0], '%Y-%m-%d_%H:%M:%S.%f')
    numbers = [float(stat) for stat in stats_row[1:]]
    return ProgressRecord(date, *numbers)


def parse_scores_row(scores_row):
    return ScoresRecord(*map(float, scores_row))


def get_sorted_summary(start_dir, glob_filename):
    stats = []
    for file in pathlib.Path(start_dir).glob(f"**/{glob_filename}"):
        stats.append((extract_csv_rows(file, parse_progress_stats_row)[0].mean_sum, file.parent))
    stats.sort(key=lambda stat: stat[0])
    return stats


def remove_all_empty_directories(path):
    if not os.path.isdir(path):
        return

    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isdir(file):
            remove_all_empty_directories(file)
    if len(os.listdir(path)) == 0:
        os.rmdir(path)


def plot_epsilon_func(c):
    plt.rcParams.update({'font.size': 18})
    x = np.arange(0, 1000)
    y = c / (c + x)
    plt.plot(x, y)
    plt.show()


def plot_smoothed_average_score(path, smooth_avg_over_n_games: int):
    all_scores = extract_csv_rows(path, parse_scores_row)
    all_scores_np = np.array(all_scores)
    games = np.arange(len(all_scores))
    avg_sums = []
    for start in range(100, len(all_scores)):
        avg_sums.append(np.mean(all_scores_np[max(0, start - smooth_avg_over_n_games):start, 4]))
    plt.plot(games[100:], avg_sums)
    plt.xlabel('numer rozgrywki')
    plt.ylabel('wynik')
    plt.show()


def generate_max_tile_histogram(path, last_n_scores=0):
    all_scores = extract_csv_rows(path, parse_scores_row)
    all_scores_np = np.array(all_scores)
    print(np.mean(all_scores_np[-1000:,4]))
    return np.unique(all_scores_np[-last_n_scores:,3], return_counts=True)

