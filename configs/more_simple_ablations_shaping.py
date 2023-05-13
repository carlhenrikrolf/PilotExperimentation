from .envs import cells3hard_env as env
from agents import PeUcrlAgt, NoPruningAgt, NoShieldAgt, UnsafeBaselineAgt

repeats = 20

config = {
    'env': [
        env
    ] * 4 * repeats,
    'agt': [
        PeUcrlAgt,
        NoPruningAgt,
        NoShieldAgt,
        UnsafeBaselineAgt,
    ] * 1 * repeats,
    'seed': [
        None
    ] * 4 * repeats,
    'regulatory_constraints': [
        'P>=1 [ G n<=2 ]',
    ] * 4 * repeats,
    'max_n_time_steps': [
        400,
    ] * 4 * repeats,
    'dir': [str(i) + '/' for i in range(4 * repeats)],
    'super_dir': 'more_simple_ablations_shaping/',
    'max_workers': 9,
}

assert len(config['env']) == len(config['agt']) == len(config['seed']) == len(config['regulatory_constraints']) == len(config['max_n_time_steps']) == len(config['dir'])