# settings
agent_id_set = ['peucrl_minus_r', 'peucrl_minus_r']
config_file_set = ['polarisation_11.json', 'polarisation_12.json']
experiment_dir_set = ['par_1', 'par_2']

# import modules
from utils import train

from concurrent.futures import ProcessPoolExecutor
from os import system

# make checks
assert len(agent_id_set) == len(config_file_set) == len(experiment_dir_set)
system('cd results; rm -r -f par_1; rm -r -f par_2')

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
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(parallel_train, agent_id_set, config_file_set, experiment_dir_set)

if __name__ == '__main__':
    main()