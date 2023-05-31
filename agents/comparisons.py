from .peucrl import PeUcrlAgt

import numpy as np

class PsoAlwaysSafeAgt(PeUcrlAgt):

    def name(self):
        return 'PSO/AlwaysSafe'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        assert type(regulatory_constraints) is list
        assert set(regulatory_constraints).issubset(set(self.prior_knowledge.cell_classes))

    def verify(self):
        if cell is in regulatory_constraints:
            return True
        else:
            return False

class eGreedyAgt(PeUcrlAgt):

    def name(self):
        return 'e-Greedy'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        assert type(regulatory_constraints) is function

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        for cell in range(self.prior_knowledge.n_cells):
            sample x
            if x >= self.regulatory_constraints(self.t):
                choose random policy

class BoundedDivergenceAgt(PeUcrlAgt):
    
    def name(self):
        return 'Bounded divergence'
    
    # accepts no regulatory constraints.
    # accepts an upper bound to the divergence
    # {'KL': 0.4, 'SED': 0.2}

class AupAgent(PeUcrlAgt):

    def name(self):
        return 'AUP'

    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        assert type(regulatory_constraints) is dict
        assert 'penalty_factor' in regulatory_constraints
        assert 'n_aux_reward_funcs' in regulatory_constraints
        self.aux_reward_funcs = np.random.rand(shape=(self.prior_knowledge.n_states,self.prior_knowledge.n_actions,regulatory_constraints['n_aux_reward_funcs']))
    
    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        return target_policy
    
    def reward_shaping(self, tabular_state, tabular_action):
        standard = super().reward_shaping(tabular_state, tabular_action)
        penalty = self.regulatory_constraints['penalty_factor'] * sum(
            [ self.aux_reward_func[tabular_state, tabular_action, index] for index in range(self.regulatory_constraints['n_aux_reward_funcs']) ]
        )
        return standard - penalty

