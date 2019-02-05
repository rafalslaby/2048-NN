import argparse

parser = argparse.ArgumentParser(description='''Neural network playing 2048
See README.md for more.
''')

valid_actions = ['results_summary', 'train', 'watch_best', 'find_best']
parser.add_argument('action', choices=valid_actions)
parser.add_argument('--render', action='store_true')
parser.add_argument('--out_dir', default='results', type=str)
parser.add_argument('--fps', help='game animation moves per second', type=int, default=2)
parser.add_argument('--dir', help='directory for find_best and results_summary searches', type=str, default='results')
parser.add_argument('-v', help='save statistics', action='store_true')

args = parser.parse_args()

from training import *
from results_analysis import *

if args.action == 'results_summary':
    stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
    print(*stats, sep='\n')

elif args.action == 'train':
    start_training(args.out_dir, args.render, args.fps)
elif args.action == 'watch_best':
    stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
    conf_dir = stats[-1][1]
    print(conf_dir)
    examine_model(pathlib.Path(conf_dir), games=1000000, render=True, render_fps=args.fps, verbose=args.v)
elif args.action == 'find_best':
    stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
    conf_dir = stats[-1][1]
    print(conf_dir)
