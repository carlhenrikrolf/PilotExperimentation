from .envs import deadlock_env_set
# ablations
from agents import PeUcrlAgt, NoShieldAgt, NoPruningAgt, UnsafeBaselineAgt

n_repeats = 25
n_agts = 4

config = {
    'env': deadlock_env_set,
    'agt': [
	PeUcrlAgt,
        NoShieldAgt,
        NoPruningAgt,
        UnsafeBaselineAgt,
    ] * n_repeats,
    'seed': [i for i in range(n_agts * n_repeats)],
    'regulatory_constraints': [
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'prism_props': 'N/A'},
        {'prism_props': 'P>=0.5 [ X X n<=1 ]'},
        {'prism_props': 'N/A'},
    ] * n_repeats,
    'max_n_time_steps': [int(1e3)] * n_agts * n_repeats,
    'dir': [str(dir) + '/' for dir in range(n_agts * n_repeats)],
    'super_dir': 'deadlock_ablations/',
    'max_workers': 50,
}

