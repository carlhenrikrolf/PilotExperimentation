from .peucrl import PeUcrlAgt
from agents.utils import *
from gym_cellular.envs.utils import generalized_cellular2tabular as cellular2tabular, generalized_tabular2cellular as tabular2cellular

import copy as cp
import numpy as np


class AlwaysSafeAgtPsoAgt(PeUcrlAgt):

    """This agent implements both the AlwaysSafe algorithm (Simao et al.) and the Path-Specific Objectives (PSO) algorithm (Farquhar et al.).
    It implements both because they reduce to the same algorithm in the cellular MDP setting.
    The regulatory constraints is a set of cell classes for which the corresponding intracellular policies are safe to update.
    In AlwaysSafe, this corresponds to the safaety-relevant variables.
    In PSO, this corresponds to delicate states.
    The shield works by updating all cells that are not delicate.
    Otherwise, the algorithm is an adaptation of the UCRL2 algorithm for cellular MDPs.
    It does not use the prior knowledge that cells have identical transition functions as this is neither assumed in AlwaysSafe nor in PSO.
    """

    def name(self):
        return 'AlwaysSafe/PSO'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        super().__init__(seed,prior_knowledge,{'prism_props': 'none'})
        self.prior_knowledge.identical_intracellular_transitions = False
        # check input
        self.regulatory_constraints = regulatory_constraints # e.g. {'delicate_cell_classes': ['children']}
        delicate_cell_classes = self.regulatory_constraints['delicate_cell_classes']
        assert type(delicate_cell_classes) is list
        assert set(delicate_cell_classes).issubset(set(self.prior_knowledge.cell_classes))
        # transform into cell indices using prior knowledge
        self.delicate_cell_set = set()
        for cell in range(self.prior_knowledge.n_cells):
            for count, cell_class in enumerate(self.prior_knowledge.cell_classes):
                if count in self.prior_knowledge.cell_labelling[cell]:
                    if cell_class in delicate_cell_classes:
                        self.delicate_cell_set.add(cell)

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set)) # randomisation not relevant here
            cell_set -= {cell}
            if cell not in self.delicate_cell_set:
                tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
                self.policy_update[cell] = 1 # only relevant for debugging here
                self.data['updated_cells'] = self.data['updated_cells'] + str(cell) + '|'
        self.policy = cp.copy(tmp_policy)


class NationLikeAgt(PeUcrlAgt):

    """This agent is a simple model of nations (or similar jurisdictions) that do not coordinate their regulations.
    Rather they tend to be conservative and only update their policies sporadically.
    When they do so, they do so greedily rather than according to some exploratory policy.
    Safety comes as a consequence of this decentralisation.
    If one nation (or other jurisdiction) makes a mistake, the other will learn.
    The regulatory constraints contain two features:
    - conservativeness: a float or a list of floats between 0 and 1 that determines the probability of updating a policy greedily
    - update_frequency: an integer that determines how often a policy is updated
    It does not use the prior knowledge that cells have identical transition functions.
    """

    def name(self):
        return 'nation-like'
    
    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        super().__init__(seed,prior_knowledge,{'prism_props': 'none'})
        self.prior_knowledge.identical_intracellular_transitions = False
        self.regulatory_constraints = regulatory_constraints # e.g. {'conservativeness': 0.2, 'update_frequency': 50}
        self.conservativeness = self.regulatory_constraints['conservativeness']
        if type(self.conservativeness) is float:
            self.conservativeness = [self.conservativeness] * self.prior_knowledge.n_cells
        elif type(self.conservativeness) is not list:
            raise ValueError('update_frequency must be a float or a list of floats')
        assert (np.array(self.conservativeness) <= 1).all() and (0 <= np.array(self.conservativeness)).all()
        self.update_frequency = self.regulatory_constraints['update_frequency']
        assert 0 <= self.update_frequency

    def stopping_criterion(self):
        if self.t % self.update_frequency == 0:
            self.new_episode = True
        else:
            self.new_episode = False

    def distances(self):
        # greediness means zero distances
        assert (self.p_distances == 0).all()
        assert (self.r_distances == 0).all()

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            # with small probability, update the policy greedily
            if np.random.rand() < self.conservativeness[cell]:
                tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
        self.policy = cp.copy(tmp_policy)


class AupAgt(PeUcrlAgt):

    """This aget uses the same regularizer as AUP although the RL aglorithm is based on cellular UCRL2."""

    def name(self):
        return 'AUP'

    def __init__(self,seed,prior_knowledge,regulatory_constraints):
        super().__init__(seed,prior_knowledge,{'prism_props': 'none'})
        self.prior_knowledge.identical_intracellular_transitions = False
        self.regulatory_constraints = regulatory_constraints # e.g. {'regularization_param': 1., 'n_aux_reward_funcs': 10}
        self.regularization_param = self.regulatory_constraints['regularization_param']
        assert 0 <= self.regularization_param
        self.n_aux_reward_funcs = self.regulatory_constraints['n_aux_reward_funcs']
        assert 0 <= self.n_aux_reward_funcs
        # initialise auxiliary reward functions randomly according to seed
        self.aux_reward_funcs = np.random.rand(
            self.prior_knowledge.n_states,
            self.prior_knowledge.n_actions,
            self.n_aux_reward_funcs,
        )

    
    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        # no shielding
        return target_policy
    

    def auxVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=int(1e6)): # max_iter=1000
        
        """This function implements a standard value iterationn algorithm, i.e. not an extended one.
        The regulatory constraints should accept n_aux_reward_funcs as an argument.
        It should also accept a regularization parameter.
        """

        # initialise a Q-map, which is not necessary in PeUcrl
        Q = np.zeros(
            shape=(
                self.prior_knowledge.n_states,
                self.prior_knowledge.n_actions,
            )
        )
        # consider unsampled estimates to be from the uniform distribution
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                if (p_estimate[s, a, :] == 0).all():
                    p_estimate[s, a, :] = 1. / self.prior_knowledge.n_states
    
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.prior_knowledge.n_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.prior_knowledge.n_states):

                temp = np.zeros(self.prior_knowledge.n_actions)
                for a in range(self.prior_knowledge.n_actions):
                    # simpler standard VI
                    temp[a] = r_estimate[s, a] + sum(
                        [P * V for P, V in zip(p_estimate[s,a,:], u0[:])]
                    )
                    # let the pruning of actions still apply
                    temp[a] *= self.transition_indicator[s, a]
                    Q[s,a] = temp[a] # save to the Q-map
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.prior_knowledge.n_states)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break

            # the outputs are also different from the standard VI
            return Q

    
    def reward_shaping(self, tabular_state, tabular_action):
        standard = super().reward_shaping(tabular_state, tabular_action)
        penalty = 0
        scale = 0
        for count in range(self.n_aux_reward_funcs):
            r_estimate = self.aux_reward_funcs[:,:,count]
            p_estimate = self.p_estimate
            Q = self.auxVI(r_estimate, p_estimate)
            noop = cellular2tabular(
                self.initial_policy[:, tabular_state],
                self.prior_knowledge.action_space,
            )
            penalty += np.abs(
                Q[tabular_state,tabular_action] - Q[tabular_state,noop]
            )
            scale += Q[tabular_state,noop]
        regularizer = self.regularization_param * penalty / scale
        return standard - regularizer

