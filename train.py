# argument
filename = 'convergence_debug'

# import modules
from utils.save import initialize_save, initialize_data
from utils.train import instantiate, train

import argparse
import os
from importlib import import_module
import pickle

# Parse arguments
parser = argparse.ArgumentParser(description='Run training')
parser.add_argument(
    'filename',
    metavar='<filename>',
    help="Runs training according to configurations in 'config/<filename>.py'",
)
parser.add_argument(
    '--continue',
    default=None,
    dest='extra_time_steps',
    metavar='t',
    help='Continue training from a previous checkpoint for t number of extra time steps',
)
args = parser.parse_args()
try:
    config_module = args.filename
except:
    config_module = filename

# import config
config_module = 'configs.' + config_module
config = import_module(config_module).config

# train
if args.extra_time_steps is None:
    path = initialize_save(config)
else:
    path = 'results/' + config['dir']

if type(path) is str:

    kwargs = config['kwargs']
    if args.extra_time_steps is None:
        env, agt = instantiate(config)
        max_n_time_steps = config['max_n_time_steps']
        initialize_data(path,**kwargs)
        restart = False
    else:
        with open(path + 'backup.pkl', 'rb') as backup_file:
            backup = pickle.load(backup_file)
        env = backup['env']
        agt = backup['agt']
        max_n_time_steps = int(args.extra_time_steps)
        restart = True
    train(path,env,agt,max_n_time_steps,restart=restart,**kwargs)

else: # multiple runs

    if args.extra_time_steps is not None:
        raise RuntimeError('Option --continue not yet supported for multiple runs')

    n = len(path)
    env = [None] * n
    agt = [None] * n
    max_n_time_steps = [None] * n
    for i in range(n):
        env[i], agt[i] = instantiate(config, index=i)
        max_n_time_steps[i] = config['max_n_time_steps'][i]
        initialize_data(path[i])

    if config['max_workers'] >= 2: # parallel
        from concurrent.futures import ProcessPoolExecutor
        def parallel_train(path, env, agt, max_n_time_steps):
            try:
                train(path, env, agt, max_n_time_steps)
            except BaseException as error:
                with open(path + 'error.txt', 'a') as error_file:
                    error_file.write(str(error))
        try:
            def main():
                with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
                    executor.map(parallel_train, path, env, agt, max_n_time_steps)
            if __name__ == '__main__':
                main()
        except BaseException as error:
            with open(config['super_dir'] + 'error.txt', 'a') as error_file:
                error_file.write(str(error))

    else: # sequential
        for i in range(n):
            train(path[i],env[i],agt[i],max_n_time_steps[i])

