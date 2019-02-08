import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Neural network playing 2048
    See README.md for more.
    ''')

    valid_actions = ['results_summary', 'train', 'watch_best', 'find_best']
    parser.add_argument('action', choices=valid_actions)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--out_dir', '-o', default='results', type=str)
    parser.add_argument('--fps', help='game animation moves per second', type=int, default=5)
    parser.add_argument('--dir', '-d', help='directory for find_best and results_summary searches', type=str,
                        default='results')
    parser.add_argument('--verbose', '-v', help='save statistics', action='store_true')
    parser.add_argument('--game_over_sleep', '-s', help='time in ms to wait after finished one game', type=int,
                        default=1000)
    parser.add_argument('--jobs', '-j', type=int, default=1)

    args = parser.parse_args()

    from training import *
    from results_analysis import *

    if args.action == 'results_summary':
        stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
        print(*stats, sep='\n')

    elif args.action == 'train':
        start_training_multiprocess(args.out_dir, args.render, args.fps, args.game_over_sleep, jobs=args.jobs)
    elif args.action == 'watch_best':
        stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
        if len(stats) > 0:
            conf_dir = stats[-1][1]
            print(f"Average sum last 100 games {stats[-1][0]}")
            print(conf_dir)
            examine_model(pathlib.Path(conf_dir), games=1000000, render=True, render_fps=args.fps, verbose=args.verbose,
                          game_over_sleep=args.game_over_sleep)
        else:
            print(f"No done configurations found in {args.dir}")
    elif args.action == 'find_best':
        stats = get_sorted_summary(args.dir, 'evaluate_done.csv')
        if len(stats) > 0:
            conf_dir = stats[-1][1]
            print(f"Average sum last 100 games {stats[-1][0]}")
            print(conf_dir)
        else:
            print(f"No done configurations found in {args.dir}")
