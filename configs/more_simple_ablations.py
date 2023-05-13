from .envs import cells3easier_env as env
from agents import PeUcrlAgt, NoPruningAgt, NoShieldAgt, UnsafeBaselineAgt

repeats = 20

config = {
    'env': [
        env
    ] * 8 * repeats,
    'agt': [
        PeUcrlAgt,
        NoPruningAgt,
        NoShieldAgt,
        UnsafeBaselineAgt,
    ] * 2 * repeats,
    'seed': [
        None
    ] * 8 * repeats,
    'regulatory_constraints': [
        *['P>=1 [ G n<=2 ] & P>=1 [ G n_children<=0]'] * 4,
        *['P>=1 [ G n<=2 ] & P>=1 [ G n_children<=0] & P>=0.80 [ F<=20 n<=1 ]'] * 4,
    ] * repeats,
    'max_n_time_steps': [
        40,
    ] * 8 * repeats,
    'dir': [str(i) + '/' for i in range(8 * repeats)],
    'super_dir': 'more_simple_ablations/',
    'max_workers': 9,
}
