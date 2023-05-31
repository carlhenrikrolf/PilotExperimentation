from .envs import debug_env as env
from agents import PeUcrlAgt as Agt

config = {
    'env': env,
    'agt': Agt,
    'seed': None,
    'regulatory_constraints': 'P>=0.5 [ X X n<=2 ]',
    'max_n_time_steps': int(1e8),
    'dir': 'debug/',
    'kwargs': {},
}

