from .envs import debug_env as env
from agents import AupAgt as Agt

config = {
    'env': env,
    'agt': Agt,
    'seed': None,
    'regulatory_constraints': {'regularization_param': 1., 'n_aux_reward_funcs': 10},
    'max_n_time_steps': int(1e3),
    'dir': 'debug/',
    'kwargs': {},
}

