from .envs import reset_env_set
# ablations
from agents import PeUcrlAgt, UnsafeBaselineAgt
# sota
from agents import AlwaysSafeAgtPsoAgt, AupAgt, NationLikeAgt

n_repeats = 3
n_agts = 5

config = {
    'env': reset_env_set,
    'agt': [
        PeUcrlAgt,
        UnsafeBaselineAgt,
        AlwaysSafeAgtPsoAgt,
        AupAgt,
        NationLikeAgt,
    ] * n_repeats,
    'seed': [i for i in range(n_agts * n_repeats)],
    'regulatory_constraints': [
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'delicate_cell_classes': ['children']},
        {'regularization_param': 1., 'n_aux_reward_funcs': 10},
        {'conservativeness': 0.2, 'update_frequency': 50},
    ] * n_repeats,
    'max_n_time_steps': [int(1e3)] * n_agts * n_repeats,
    'dir': [str(dir) + '/' for dir in range(n_agts * n_repeats)],
    'super_dir': 'reset/',
    'max_workers': 50,
}
