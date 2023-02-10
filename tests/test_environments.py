import gymnasium as gym
import gym_cellular
import numpy as np

class Test_Environments:

    def test_polarisation_instantiation(self):
        
        env = gym.make('gym_cellular/Polarisation-v1')
        assert isinstance(env, gym.Env)

    def test_polarisation_reset(self):
        
        env = gym.make('gym_cellular/Polarisation-v1')
        state, info = env.reset()
        assert type(state) is dict
        assert type(info) is dict
        assert type(env.cellular_encoding(state)) is np.ndarray
    
    def test_polarisation_step(self):
            
        env = gym.make('gym_cellular/Polarisation-v1')
        _, _ = env.reset()
        action = np.zeros(env.n_users, dtype=int)
        current_state, reward, terminated, truncated, info = env.step(action)
        assert type(current_state) is dict
        assert type(reward) is np.float64
        assert type(terminated) is bool
        assert type(truncated) is bool
        assert type(info) is dict
        assert type(env.cellular_decoding(action)) is np.ndarray