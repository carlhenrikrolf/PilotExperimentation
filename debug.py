import os
import gymnasium as gym

from train import debug
from agents import *
os.system('pip3 install -e gym-cellular -q')
import gym_cellular

env_id = [
    'gym_cellular/Cells3States3Actions3-v0',
    'gym_cellular/Debug-v0',
]
path = 'results/debug/'
env_config = {
    'args': [
        env_id[1], # id
        None, # max_episode_steps
        False, # autoreset
        None, # apply_api_compatibility
        None, # disable_env_checker
    ],
    'kwargs': {
        'difficulty': 'hard',
    },
}
regulatory_constraints = [
    'true', #0
    'P>=1 [ G n <= 1 ]', #1
    'P>=0.8 [ F<=5 n <= 1]', #2
    'P>=1 [ G n_children<=0 ] & P>=0.98 [ F<=50 n<=2]', #3
    'P>=0.7 [ G n<=2]', #4
]
config = {
    'env': env_config,
    'agt': PeUcrlAgt,
    'seed': 1119858,
    'regulatory_constraints': regulatory_constraints[0],
    'max_n_time_steps': 100000,
}

env = gym.make(
    *config['env']['args'],
    **config['env']['kwargs'],
)
Agt = config['agt']
agt = Agt(
    seed=config['seed'],
    prior_knowledge=env.prior_knowledge,
    regulatory_constraints=config['regulatory_constraints'],
)

debug(
    path,
    env,
    agt,
    config['max_n_time_steps'],
)