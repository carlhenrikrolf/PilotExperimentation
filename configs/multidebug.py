from .envs import cells3easy_env
from agents import AlwaysSafeAgtPsoAgt, NationLikeAgt, AupAgt

config = {
    'env': [cells3easy_env] * 9,
    'agt': [
        AlwaysSafeAgtPsoAgt,
        AlwaysSafeAgtPsoAgt,
        AlwaysSafeAgtPsoAgt,
        NationLikeAgt,
        NationLikeAgt,
        NationLikeAgt,
        AupAgt,
        AupAgt,
        AupAgt,
    ],
    'seed': [None] * 9,
    'regulatory_constraints': [
        {'delicate_cell_classes': ['children']},
        {'delicate_cell_classes': ['children']},
        {'delicate_cell_classes': ['children']},
        {'conservativeness': 0.2, 'update_frequency': 50},
        {'conservativeness': 0.2, 'update_frequency': 50},
        {'conservativeness': 0.2, 'update_frequency': 50},
        {'regularization_param': 1., 'n_aux_reward_funcs': 10},
        {'regularization_param': 1., 'n_aux_reward_funcs': 10},
        {'regularization_param': 1., 'n_aux_reward_funcs': 10},
    ],
    'max_n_time_steps': [int(1e6)] * 9,
    'dir': [
        'always0/',
        'always1/',
        'always2/',
        'nation0/',
        'nation1/',
        'nation2/',
        'aup0/',
        'aup1/',
        'aup2/',
    ],
    'super_dir': 'multidebug/',
    'max_workers': 9,
}
