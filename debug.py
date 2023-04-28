import os
import gymnasium as gym

from train import debug
from agents import DebugUcrl2Agt
os.system('pip3 install -e gym-cellular -q')
import gym_cellular

config = {
    'gym_id': 'gym_cellular/Debug-v0'
}
path = 'results/debug/'

env = gym.make(
    config['gym_id'],
    **config,
)
agt = DebugUcrl2Agt(
    seed=100,
    prior_knowledge=env.prior_knowledge,
    regulatory_constraints='true',
)
max_n_time_steps = 2000

debug(path, env, agt, max_n_time_steps)