# settings
agent_id_set = ['peucrl', 'peucrl', 'peucrl', 'peucrl', 'peucrl', 'ucrl2', 'ucrl2', 'ucrl2', 'ucrl2', 'ucrl2']
config_file_set = ['polarisation_0.json', 'polarisation_1.json', 'polarisation_2.json', 'polarisation_3.json', 'polarisation_4.json', 'polarisation_0.json', 'polarisation_1.json', 'polarisation_2.json', 'polarisation_3.json', 'polarisation_4.json']
experiment_dir_set = ['0peucrl', '1peucrl', '2peucrl', '3peucrl', '4peucrl', '0ucrl2', '1ucrl2', '2ucrl2', '3ucrl2', '4ucrl2']

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