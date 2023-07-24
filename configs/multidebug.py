from .envs import debug_env_set
from agents import PeUcrlAgt

config = {
    'env': debug_env_set,
    'agt': [
        PeUcrlAgt
    ] * 3,
    'seed': [0 for i in range(3)],
    'regulatory_constraints': [
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'}
    ] * 3,
    'max_n_time_steps': [int(1e2)] * 3,
    'dir': [
        str(i) + '/' for i in range(3)
    ],
    'super_dir': 'multidebug/',
    'max_workers': 9,
}
