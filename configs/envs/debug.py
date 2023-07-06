debug_env = {
    'args': [
        'gym_cellular/Cells3ResetVDeadlock-v0', # id
        None, # max_episode_steps
        False, # autoreset
        None, # apply_api_compatibility
        None, # disable_env_checker
    ],
    'kwargs': {
        'difficulty': 'easy',
        'reward_func_is_known': True,
        'deadlock': True,
        'env_seed': 0,
    }
}