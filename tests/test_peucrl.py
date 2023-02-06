from ..agents import PeUcrlAgent
import numpy as np

class Test_PeUcrl:

    def test_instantiation(self):
        
        confidence_level = 0.95
        accuracy = 0.90
        n_cells = 2
        self.n_cells = n_cells
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


        self.agt = PeUcrlAgent(
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
    
    def test_sample_action(self):
        
        self.test_instantiation()
        state = np.zeros(self.n_cells, dtype=int)
        action = self.agt.sample_action(state)
        assert action.shape == (self.n_cells,)
        

