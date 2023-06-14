from .peucrl import PeUcrlAgt, cellular2tabular, tabular2cellular

import copy as cp
import numpy as np

# maybe I want to modify side effects such that it is more of a scalar rather than a matrix?

class AlwaysSafeAgtPsoAgt(PeUcrlAgt):

    def name(self):
        return 'AlwaysSafe/PSO'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        delicate_cell_classes = eval(regulatory_constraints)
        assert type(delicate_cell_classes) is list
        assert set(delicate_cell_classes).issubset(set(self.prior_knowledge.cell_classes))
        self.delicate_cell_set = set()
        for cell in range(self.prior_knowledge.n_cells):
            for count, cell_class in enumerate(self.prior_knowledge.cell_classes):
                if cell in self.prior_knowledge.cell_labelling[count]:
                    if cell_class in delicate_cell_classes:
                        self.delicate_cell_set.add(cell)


    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set)) # randomisation not relevant here
            cell_set -= {cell}
            if cell in self.delicate_cell_set:
                tmp_policy[cell, :] = cp.copy(behaviour_policy[cell, :])
            else:
                tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
                self.policy_update[cell] = 1 # only relevant for debugging here
                self.data['updated_cells'] = self.data['updated_cells'] + str(cell) + '|'
        self.policy = cp.copy(tmp_policy)

# this comparison is a model of jurisdictions not coordinating their policies, but learning from others when something bad happens
class eOptimisticAgt(PeUcrlAgt):

    def name(self):
        return 'e-Optimistic'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False # maybe this should be as given by the environment?
        self.epsilon = eval(regulatory_constraints)
        assert type(self.epsilon) is float

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            if np.random.rand() < self.epsilon:
                tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
        self.policy = cp.copy(tmp_policy)

# this is a more classic e-greedy system, but it makes less sense for this comparison.
# class eGreedyAgt(PeUcrlAgt):

#     def name(self):
#         return 'e-Greedy'
    
#     def __init__(self,seed,prior_knowledge,regulatory_constraints):
#         PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
#         self.prior_knowledge.identical_intracellular_transitions = False
#         self.epsilon = eval(regulatory_constraints)
#         assert type(self.epsilon) is float

#     def stopping_criterion(self):
#         self.new_episode = True

#     def distances(self):
#         pass

#     def pe_shield(self, behaviour_policy, target_policy, p_estimate):
#         tmp_policy = cp.copy(target_policy)
#         cell_set = set(range(self.prior_knowledge.n_cells))
#         while len(cell_set) >= 1:
#             cell = np.random.choice(list(cell_set))
#             cell_set -= {cell}
#             if np.random.rand() < self.epsilon:
#                 tmp_policy[cell, :] = np.random.randint(
#                     low=0,
#                     high=self.prior_knowledge.n_intracellular_actions,
#                     size=self.prior_knowledge.n_states,
#                 )
#         self.policy = cp.copy(tmp_policy)


class CappedDivergenceAgt(PeUcrlAgt):
    
    def name(self):
        return 'Capped divergence'
    
    def __init__(self, seed, prior_knowledge, regulatory_constraints):
        PeUcrlAgt().__init__(seed, prior_knowledge, regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        self.cap = eval(regulatory_constraints)
        assert self.cap >= 0
        self.last_policy = cp.copy(self.initial_policy)

    def off_policy(self):
        PeUcrlAgt().off_policy()
        self.last_policy = cp.copy(self.policy)

    def reward_shaping(self, tabular_state, tabular_action):
        standard = PeUcrlAgt().reward_shaping(tabular_state, tabular_action)
        def SED(tabular_state, tabular_action):
            output = 0
            for s in range(self.prior_knowledge.n_states):
                if s == tabular_state:
                    continue
                for cell in range(self.prior_knowledge.n_cells):
                    if self.last_policy[cell, s] != self.policy[cell, s]:
                        output += 1
                        break
            cellular_action = tabular2cellular(tabular_action, self.prior_knowledge.action_space)
            for cell in range(self.prior_knowledge.n_cells):
                if self.last_policy[cell, tabular_state] != cellular_action[cell]:
                    output += 1
                    break
        penalty = max(0, SED(tabular_state, tabular_action) - self.cap)
        return standard - penalty
    


class AupAgent(PeUcrlAgt):

    def name(self):
        return 'AUP'

    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        PeUcrlAgt().__init__(seed,prior_knowledge,regulatory_constraints)
        self.prior_knowledge.identical_intracellular_transitions = False
        self.aux_specs = eval(regulatory_constraints)
        assert type(self.aux_specs) is dict
        assert 'penalty_factor' in self.aux_specs
        assert 'n_aux_reward_funcs' in self.aux_specs
        self.aux_reward_funcs = np.random.rand(
            shape=(
                self.prior_knowledge.n_states,
                self.prior_knowledge.n_actions,
                self.aux_specs['n_aux_reward_funcs']
            )
        )
    
    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        return target_policy
    

    def auxVI(self, r_estimate, p_estimate, policy, epsilon=0.01, max_iter=int(1e6), kind='q'): # max_iter=1000
        Q = np.zeros(
            shape=(
                self.prior_knowledge.n_states,
                self.prior_knowledge.n_actions,
            )
        )
        u0 = self.u - min(self.u)  #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.prior_knowledge.n_states)
        sorted_indices = np.arange(self.prior_knowledge.n_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.prior_knowledge.n_states):

                temp = np.zeros(self.prior_knowledge.n_actions)
                for a in range(self.prior_knowledge.n_actions):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    if hasattr(self.prior_knowledge, 'reward_func'):
                        optimistic_reward = self.reward_func[s, a]
                    else:
                        optimistic_reward = min([1, r_estimate[s, a] + self.r_distances[s, a]])
                    optimistic_reward = min([1, optimistic_reward + self.reward_shaping(s, a)]) # I think this should work fine theoretically for nown reward functions, but it is a bit less clear what I should do with unknown reward functions, I might get a factor of 2 somewhere in the proof.
                    temp[a] = optimistic_reward + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                    temp[a] *= self.transition_indicator[s, a]
                    Q[s,a] = temp[a] # change here and below
                tabular_action = cellular2tabular(
                    policy[:, s],
                    self.prior_knowledge.action_space,
                )
                u1[s] = temp[tabular_action]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.prior_knowledge.n_states)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break
            if kind == 'q':
                return Q
            elif kind == 'u':
                return u1
            else:
                raise ValueError('kind must be either q or u')

    
    def reward_shaping(self, tabular_state, tabular_action):
        standard = super().reward_shaping(tabular_state, tabular_action)
        Q = np.zeros(
            shape=(
                self.prior_knowledge.n_states,
                self.prior_knowledge.n_actions,
                self.aux_specs['n_aux_reward_funcs'],
            )
        )
        penalty = 0
        for count in range(self.aux_specs['n_aux_reward_funcs']):
            Q[:,:,count] = self.auxVI(self.r_estimate, self.p_estimate, self.policy)
            penalty += np.abs(Q[tabular_state,tabular_action,count] - self.uinit[tabular_state])
        scale = ... # uninit must be on every reward function
        
        return standard - penalty

