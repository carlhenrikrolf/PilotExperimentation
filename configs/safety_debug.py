from .envs import debug_env
from agents import PeUcrlAgt

config = {
    'env': debug_env,
    'agt': PeUcrlAgt,
    'seed': 0,
    'regulatory_constraints': 'P>=1 [ G n<=1 ]',
    'max_n_time_steps': 100,
    'dir': 'debug/',
}

