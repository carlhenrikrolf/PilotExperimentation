from .envs import cells3easy_env as env
from agents import PeUcrlAgt

config = {
    'env': env,
    'agt': PeUcrlAgt,
    'seed': 0,
    'regulatory_constraints': 'true',
    'max_n_time_steps': 100,
    'dir': 'debug/',
}

