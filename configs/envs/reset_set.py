n_repeats = 25
n_agents = 5
reset_env_set = [
    {
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
            'deadlock': False,
            'env_seed': i,
        }
    } for i in range(n_repeats * n_agents)
]