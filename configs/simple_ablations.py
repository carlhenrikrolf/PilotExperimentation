from .envs import cells3easy_env
from agents import PeUcrlAgt, NoPruningAgt, NoShieldAgt

config = {
    'env': [cells3easy_env] * 9,
    'agt': [
        PeUcrlAgt,
        PeUcrlAgt,
        PeUcrlAgt,
        NoPruningAgt,
        NoPruningAgt,
        NoPruningAgt,
        NoShieldAgt,
        NoShieldAgt,
        NoShieldAgt,
    ],
    'seed': [None] * 9,
    'regulatory_constraints': ['P>=1 [ G n<=2 ] & P>=1 [ G n_children<=0] & P>=0.80 [ F<=20 n<=1 ]'] * 9,
    'max_n_time_steps': [50] * 9,
    'dir': [
        'peucrl0/',
        'peucrl1/',
        'peucrl2/',
        'nopruning0/',
        'nopruning1/',
        'nopruning2/',
        'noshield0/',
        'noshield1/',
        'noshield2/',
    ],
    'super_dir': 'simple_ablations/',
    'max_workers': 9,
}
