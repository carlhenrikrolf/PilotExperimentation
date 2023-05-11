from gym_cellular.envs.cells3states3actions3 import right_polarizing, multiple_optima, nonlinear

cells3easier_env = {
    'args': [
        'gym_cellular/Cells3States3Actions3-v0', # id
        None, # max_episode_steps
        False, # autoreset
        None, # apply_api_compatibility
        None, # disable_env_checker
    ],
    'kwargs': {
        'difficulty': 'easy',
        'reward_func': nonlinear,
        'identical_intracellular_transitions': True,
    }
}