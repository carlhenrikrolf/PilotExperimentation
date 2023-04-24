n_extra_time_steps = 20000000
experiment_dir_set = ['0peucrl', '1peucrl', '2peucrl', '0noshielding', '1noshielding', '2noshielding', '0noshieldingnopruning', '1noshieldingnopruning', '2noshieldingnopruning']

from utils import restart_train

from concurrent.futures import ProcessPoolExecutor
from os import system

def restart_parallel_train(experiment_dir, n_extra_time_steps):
    restart_train(
        experiment_dir=experiment_dir,
        n_extra_time_steps=n_extra_time_steps,
        quiet=True,
    )

n_sxtra_time_steps_set  = [n_extra_time_steps for _ in experiment_dir_set]

def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(restart_parallel_train, experiment_dir_set, n_sxtra_time_steps_set)

if __name__ == '__main__':
    main()