from .envs import cells3hard_env as env
from agents import PeUcrlAgt as Agt

config = {
    'env': env,
    'agt': Agt,
    'seed': None,
    'regulatory_constraints': 'P>=1 [ G n<=2 ] & P>=1 [ G n_children<=0]',
    'max_n_time_steps': 400, #100000
    'dir': 'debug/',
}

