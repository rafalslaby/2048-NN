import os, csv, datetime, sys
import pathlib
from helper_functions import *

# RScore = namedtuple('RScore', 'sum max_tile')
StatsRow = namedtuple('ProgressRecord',
                      ['time', 'step', 'game', 'eps', 'max_step', 'max_tile', 'max_sum', 'max_rew', 'mean_step',
                       'mean_tile', 'mean_sum', 'mean_rew', 'max_100_step', 'max_100_tile', 'max_100_sum',
                       'max_100_rew', 'mean_100_step', 'mean_100_tile', 'mean_100_sum', 'mean_100_rew'])


def iter_dirs_with_file(filename, specific_dir=None):
    path = pathlib.Path(specific_dir) if specific_dir else pathlib.Path('.')
    for file in path.glob(f"**/{filename}"):
        if file.stat().st_size > 0:
            yield file.parent


def extract_stats(path):
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header
        return [parse_stats_strings(row) for row in reader]


def parse_stats_strings(stats_row):
    date = datetime.datetime.strptime(stats_row[0], '%Y-%m-%d_%H:%M:%S.%f')
    numbers = [float(stat) for stat in stats_row[1:]]
    return StatsRow(date, *numbers)


def get_sorted_summary(start_dir, glob_filename):
    stats = []
    for file in pathlib.Path(start_dir).glob(f"**/{glob_filename}"):
        stats.append((extract_stats(file)[0].mean_sum, file.parent))
    stats.sort(key=lambda stat: stat[0])
    return stats


get_sorted_summary(pathlib.Path('results'), 'evaluate_done.csv')



def remove_all_empty_directories(path):
    if not os.path.isdir(path):
        return

    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isdir(file):
            remove_all_empty_directories(file)
    if len(os.listdir(path)) == 0:
        os.rmdir(path)


