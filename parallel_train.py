# settings
agent_id_set = ['peucrl', 'peucrl', 'peucrl', 'noshielding', 'noshielding','noshielding', 'noshieldingnopruning', 'noshieldingnopruning', 'noshieldingnopruning']
config_file_set = ['polarisation_5.json', 'polarisation_6.json', 'polarisation_7.json','polarisation_5.json', 'polarisation_6.json', 'polarisation_7.json','polarisation_5.json', 'polarisation_6.json', 'polarisation_7.json']
experiment_dir_set = ['0peucrl', '1peucrl', '2peucrl', '0noshielding', '1noshielding', '2noshielding', '0noshieldingnopruning', '1noshieldingnopruning', '2noshieldingnopruning']

# import modules
from utils import train

from concurrent.futures import ProcessPoolExecutor
from os import system

# make checks
assert len(agent_id_set) == len(config_file_set) == len(experiment_dir_set)

system('pip3 install -e gym-cellular -q')

# run parallel training
def parallel_train(agent_id, config_file_name, experiment_dir):
    train(
        agent_id=agent_id,
        config_file_name=config_file_name,
        experiment_dir=experiment_dir,
        quiet=True,
    )

def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(parallel_train, agent_id_set, config_file_set, experiment_dir_set)

if __name__ == '__main__':
    main()