from .envs import debug_env as env
from agents import UnsafeBaselineAgt as Agt

config = {
    'env': env,
    'agt': Agt,
    'seed': None,
    'regulatory_constraints': 'P>=1 [ G n<=1 ]',
    'max_n_time_steps': 40,
    'dir': 'debug/',
    'kwargs': {'environment': 'debug'},
}

