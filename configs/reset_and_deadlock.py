from .envs import reset_env, deadlock_env
# ablations
from agents import PeUcrlAgt, NoShieldAgt, NoPruningAgt, UnsafeBaselineAgt
# sota
from agents import AlwaysSafeAgtPsoAgt, AupAgt, NationLikeAgt

n_repeats = 3

n_agts = 7
n_envs = 2

config = {
    'env': [
        *[reset_env] * n_agts,
        *[deadlock_env] * n_agts,
    ] * n_repeats,
    'agt': [
        *[
            PeUcrlAgt,
            NoShieldAgt,
            NoPruningAgt,
            UnsafeBaselineAgt,
            AlwaysSafeAgtPsoAgt,
            AupAgt,
            NationLikeAgt,
        ] * n_envs,
    ] * n_repeats,
    'seed': [None] * n_agts * n_envs * n_repeats,
    'regulatory_constraints': [
        *[
            {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
            {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
            {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
            {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
            {'delicate_cell_classes': ['children']},
            {'regularization_param': 1., 'n_aux_reward_funcs': 10},
            {'conservativeness': 0.2, 'update_frequency': 50},
        ] * n_envs,
    ] * n_repeats,
    'max_n_time_steps': [
        *[int(1e7)] * n_agts,
        *[int(1e3)] * n_agts,
    ] * n_repeats,
    'dir': [str(dir) + '/' for dir in range(n_agts * n_envs * n_repeats)],
    'super_dir': 'reset_and_deadlock/',
    'max_workers': 9,
}
