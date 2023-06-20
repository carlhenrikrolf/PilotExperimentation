from .envs import debug_env as env
from agents import AlwaysSafeAgtPsoAgt as Agt

config = {
    'env': env,
    'agt': Agt,
    'seed': None,
    'regulatory_constraints': {'delicate_cell_classes': ['children']},
    'max_n_time_steps': int(1e3),
    'dir': 'debug/',
    'kwargs': {},
}

