from ..agents import PeUcrlAgent
import numpy as np

def minimal_instantiation():

    confidence_level = 0.95
    accuracy = 0.90
    n_cells = 2
    n_cells = n_cells
    n_intracellular_actions = 2
    def cellular_encoding(state):
        return state
    n_intracellular_states = 2
    def cellular_decoding(action):
        return action
    def reward_function(state, action):
        return 0
    cell_classes = None
    cell_labelling_function = None
    regulatory_constraints = None
    initial_policy = np.zeros((n_cells, n_intracellular_states**n_cells), dtype=int) # flat states


    agt = PeUcrlAgent(
        confidence_level=confidence_level,
        accuracy=accuracy,
        n_cells=n_cells,
        n_intracellular_states=n_intracellular_states,
        cellular_encoding=cellular_encoding,
        n_intracellular_actions=n_intracellular_actions,
        cellular_decoding=cellular_decoding,
        reward_function=reward_function,
        cell_classes=cell_classes,
        cell_labelling_function=cell_labelling_function,
        regulatory_constraints=regulatory_constraints,
        initial_policy=initial_policy,
    )

    return agt


class Test_PeUcrl:

    def test_instantiation(self):
        
        agt = minimal_instantiation()
        assert isinstance(agt, PeUcrlAgent)

    
    def test_sample_action(self):
        
        agt = minimal_instantiation()
        state = np.zeros(agt.n_cells, dtype=int)
        action = agt.sample_action(state)
        assert action.shape == (agt.n_cells,)
    

    def test_flatten_is_invertible(self):

        agt = minimal_instantiation()
        
        x = np.zeros(agt.n_cells, dtype=int)
        y = agt._unflatten(flat_state=agt._flatten(state=x))
        assert (x == y).all()

        x[0] = 1
        y = agt._unflatten(flat_action=agt._flatten(action=x))
        assert (x == y).all()

        x = np.ones(agt.n_cells, dtype=int)
        y = agt._unflatten(flat_action=agt._flatten(action=x))
        assert (x == y).all()

        x[0] = 0
        y = agt._unflatten(flat_action=agt._flatten(action=x))
        assert (x == y).all()

    
    def test_initial_update_no_new_episode(self):

        agt = minimal_instantiation()
        previous_state = np.zeros(agt.n_cells, dtype=int)
        _ = agt.sample_action(previous_state)
        current_state = np.ones(agt.n_cells, dtype=int)
        reward = 0
        side_effects = np.array( # agt.n_cells == 2
            [
                ['safe', 'silent'],
                ['silent', 'unsafe'],
            ]
        )
        agt.update(current_state=current_state, reward=reward, side_effects=side_effects)
        assert (agt.current_state == current_state).all()
        assert np.sum(agt.current_episode_count) == 1
        assert np.sum(agt.cellular_current_episode_count) <= agt.time_step * agt.n_cells # <=2
        assert (agt.intracellular_episode_count <= agt.time_step * agt.n_cells).all() # <=2

    
    def test_inner_maximum(self):
        
        agt = minimal_instantiation()
        flat_state = 0
        flat_action = 0
        values = np.zeros(agt.n_states, dtype=float)
        inner_max = agt._inner_max(flat_state, flat_action, values)
        assert inner_max == 0

    

    def test_update_confidence_sets(self):
        pass


    def test_extended_value_iteration(self):
        
        agt = minimal_instantiation()
        agt._extended_value_iteration()
        assert (agt.target_policy == 0).all()


    def test_pe_shield(self):
        pass


    def test_initial_update_new_episode(self):
        pass



def polarisation_instantiation():
    pass


class Test_PeUcrl_in_PolarisationEnv:
    
    def test_instantiation(self):
        pass


    def test_rewards(self):
        pass
