# argument
filename = 'convergence_debug'

# import modules
from utils.save import initialize_save, initialize_data
from utils.train import instantiate, train

import argparse
import os
from importlib import import_module

# Parse arguments
parser = argparse.ArgumentParser(description='Run training')
parser.add_argument(
    'filename',
    metavar='<filename>',
    help="Runs training according to configurations in 'config/<filename>.py'",
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
path = initialize_save(config)

if type(path) is str:

    env, agt = instantiate(config)
    max_n_time_steps = config['max_n_time_steps']
    initialize_data(path)
    train(path,env,agt,max_n_time_steps)

else: # multiple runs

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
        def main():
            with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
                executor.map(train, path, env, agt, max_n_time_steps)
        if __name__ == '__main__':
            main()

    else: # sequential
        for i in range(n):
            train(path[i],env[i],agt[i],max_n_time_steps[i])

