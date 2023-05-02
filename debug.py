import os
import gymnasium as gym

from train import debug
from agents import *
os.system('pip3 install -e gym-cellular -q')
import gym_cellular

path = 'results/debug/'
env_config = {
    'args': [
        'gym_cellular/Debug-v0', # id
        None, # max_episode_steps
        False, # autoreset
        None, # apply_api_compatibility
        None, # disable_env_checker
    ],
    'kwargs': {
    },
}
config = {
    'env': env_config,
    'agt': PeUcrlAgt,
    'seed': 99,
    'regulatory_constraints': 'true',
    'max_n_time_steps': 5000,
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