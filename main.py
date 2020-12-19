import argparse
from utils.atari import Atari
import atari_py

available_games = list((''.join(x.capitalize() or '_' for x in word.split(
    '_')) for word in atari_py.list_games()))


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', help='Choose from available games: ' +
                        str(available_games) + '. Default is "Breakout".', default='Breakout')
    parser.add_argument(
        '-m', '--mode', help='Choose from available modes: training, testing. Default is "training".', default='training')
    parser.add_argument('-tsl', '--total_step_limit',
                        help='Choose how many total steps (frames visible by agent) should be performed. Default is 10000000.', default=10000000, type=int)
    parser.add_argument('-trl', '--total_run_limit',
                        help='Choose after how many runs we should stop. Default is None (no limit).', default=None, type=int)

    parser.add_argument(
        '-r', '--render', help='Choose if the game should be rendered. Default is False.', default=False, type=bool)
    parser.add_argument(
        '-s', '--sign_only', help='Choose whether we should clip rewards to its sign. Default is "True"', default=True, type=bool)
    args = parser.parse_args()
    game_name = args.game
    mode_name = args.mode
    total_step_limit = args.total_step_limit
    total_run_limit = args.total_run_limit
    should_render = args.render
    sign_only = args.sign_only
    print('Selected game: ' + str(game_name))
    print('Selected mode: ' + str(mode_name))
    print('Should render: ' + str(should_render))
    print('Should clip: ' + str(sign_only))
    print('Total step limit: ' + str(total_step_limit))
    print('Total run limit: ' + str(total_run_limit))
    return game_name, mode_name, total_step_limit, total_run_limit, should_render, sign_only


def main():
    game_name, mode_name, total_step_limit, total_run_limit, should_render, sign_only = parseArguments()
    atari = Atari(game_name, mode_name, total_step_limit,
                  total_run_limit, should_render, sign_only)
    atari.loop()
    atari.save()


if __name__ == '__main__':
    main()
