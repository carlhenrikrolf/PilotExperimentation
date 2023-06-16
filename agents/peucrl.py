"""Copypasted and modified, cite"""

from agents.utils import *
from gym_cellular.envs.utils import generalized_cellular2tabular as cellular2tabular, generalized_tabular2cellular as tabular2cellular

import copy as cp
import numpy as np
import os
from psutil import Process
import subprocess
import time

class PeUcrlAgt:

    def name(self):
        return 'PE-UCRL'

    def __init__(
        self,
        seed,
        prior_knowledge,
        regulatory_constraints,
    ):
        """Implementation of PeUcrl."""

        # Storing the parameters
        if seed is None:
            self.seed = int(str(time.time_ns())[-9:])
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.prior_knowledge = prior_knowledge
        self.regulatory_constraints = regulatory_constraints
        self.prism_props = regulatory_constraints['prism_props']
        assert type(self.prism_props) is str

        # Initialize counters
        self.t = 1
        self.vk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        ) #the state-action count for the current episode k
        self.Nk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        ) #the state-action count prior to episode k
        self.p_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.Pk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions, self.prior_knowledge.n_states),
            dtype=int,
        )
        self.u = np.zeros(
            shape=self.prior_knowledge.n_states,
            dtype=float,
        )
        if hasattr(prior_knowledge, 'reward_func'):
            self.reward_func = np.zeros(
                shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
                dtype=float,
            )
            for s in range(self.prior_knowledge.n_states):
                for a in range(self.prior_knowledge.n_actions):
                    self.reward_func[s, a] = self.prior_knowledge.reward_func(
                        state=self.prior_knowledge.detabularize(
                            s,
                            self.prior_knowledge.state_space,
                        ),
                        action=self.prior_knowledge.detabularize(
                            a,
                            self.prior_knowledge.action_space,
                        ),
                        next_state=(0, 0), # this needs to change
                    )
        self.r_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.Rk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        if self.prior_knowledge.identical_intracellular_transitions is True:
            self.intracellularvk = np.zeros(
                shape=(self.prior_knowledge.n_intracellular_states, self.prior_knowledge.n_intracellular_actions),
                dtype=int,
            )
            self.transfervk = np.zeros(
                shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
                dtype=int,
            )
            self.intracellularNk = np.zeros(
                shape=(self.prior_knowledge.n_intracellular_states, self.prior_knowledge.n_intracellular_actions),
                dtype=int,
            )
            assert self.intracellularNk.shape == self.intracellularvk.shape
            self.transferNk = np.zeros(
                shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
                dtype=int,
            )
            assert self.transferNk.shape == self.transfervk.shape
            self.intracellularPk = np.zeros(
                shape=(self.prior_knowledge.n_intracellular_states, self.prior_knowledge.n_intracellular_actions, self.prior_knowledge.n_intracellular_states),
                dtype=int,
            )
            assert self.intracellularPk[:,:,0].shape == self.intracellularvk.shape
        self.r_estimate = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.p_estimate = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions, self.prior_knowledge.n_states),
            dtype=float,
        )


        # Misc initializations
        initial_cellular_state = self.prior_knowledge.cellularize(
            element=self.prior_knowledge.initial_state,
            space=self.prior_knowledge.state_space,
        )
        self.last_cellular_state = initial_cellular_state
        self.initial_tabular_state = cellular2tabular(
            initial_cellular_state,
            self.prior_knowledge.state_space,
        )
        self.last_tabular_state = cp.copy(self.initial_tabular_state)
        self.current_tabular_state = cp.copy(self.initial_tabular_state)
        self.initial_policy = np.zeros(
            shape=(self.prior_knowledge.n_cells, self.prior_knowledge.n_states),
            dtype=int,
        ) # policy
        for s in range(self.prior_knowledge.n_states):
            S = self.prior_knowledge.detabularize(
                tabular_element=s,
                space=self.prior_knowledge.state_space
            )
            A = self.prior_knowledge.initial_policy(S)
            self.initial_policy[:, s] = self.prior_knowledge.cellularize(
                element=A,
                space=self.prior_knowledge.action_space,
            )
        self.policy = cp.copy(self.initial_policy)
        self.policy_update = np.zeros(
            shape=self.prior_knowledge.n_cells,
            dtype=int,
        )
        self.rc = [0.0] * self.prior_knowledge.n_cells
        self.side_effects_funcs = [{'safe', 'unsafe'} for _ in range(self.prior_knowledge.n_intracellular_states)]
        self.initial_safe_intracellular_states = set()
        for state in self.prior_knowledge.initial_safe_states:
            cellular_state = self.prior_knowledge.cellularize(
                element=state,
                space=self.prior_knowledge.state_space,
            )
            for intracellular_state in cellular_state:
                self.side_effects_funcs[intracellular_state] -= {'unsafe'}
                self.initial_safe_intracellular_states.add(intracellular_state)
        self.new_pruning = False
        self.intracellular_transition_indicator = np.ones(
            shape=(self.prior_knowledge.n_intracellular_states, self.prior_knowledge.n_intracellular_actions),
            dtype=int,
        )
        self.transition_indicator = np.ones(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        )
        self.path = [set() for _ in range(self.prior_knowledge.n_cells)]

        # data collection
        self.data = {}
        self.data['off_policy_time'] = np.nan
        self.new_episode = False
        self.new_pruning = False
        self.data['updated_cells'] = ''


    def reset_seed(self):
        np.random.seed(self.seed)


    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                self.Nk[s, a] += self.vk[s, a]

    # Auxiliary function to update v the accumulated state-action count.
    def updatev(self):
        self.vk[self.last_tabular_state, self.last_tabular_action] += 1

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.last_tabular_state, self.last_tabular_action] += self.current_reward

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.last_tabular_state, self.last_tabular_action, self.current_tabular_state] += 1

    def update_intracellularv(self):
        for si, ai in zip(self.last_cellular_state, self.last_cellular_action):
            self.intracellularvk[si, ai] += 1

    def update_intracellularP(self):
        for si, ai, next_si in zip(self.last_cellular_state, self.last_cellular_action, self.current_cellular_state):
            self.intracellularPk[si, ai, next_si] += 1

    def update_transferv(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                intracellular_pair_set = zip(
                    tabular2cellular(s, self.prior_knowledge.state_space),
                    tabular2cellular(a, self.prior_knowledge.action_space),
                )
                self.transfervk[s,a] = min(
                    [self.intracellularvk[si, ai] for si, ai in intracellular_pair_set]
                )
    
    def update_intracellularN(self):
        for si in range(self.prior_knowledge.n_intracellular_states):
            for ai in range(self.prior_knowledge.n_intracellular_actions):
                self.intracellularNk[si, ai] += self.intracellularvk[si, ai]

    def update_transferN(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                intracellular_pair_set = zip(
                    tabular2cellular(s, self.prior_knowledge.state_space),
                    tabular2cellular(a, self.prior_knowledge.action_space),
                )
                self.transferNk[s,a] = min(
                    [self.intracellularNk[si, ai] for si, ai in intracellular_pair_set]
                )


    # def update_transferv(self):
    #     for s in range(self.prior_knowledge.n_states):
    #         for a in range(self.prior_knowledge.n_actions):
    #             distinct_pair_set = set(
    #                 zip(
    #                     tabular2cellular(s, self.prior_knowledge.state_space),
    #                     tabular2cellular(a, self.prior_knowledge.action_space),
    #                 )
    #             )
    #             last_pair_list = list(
    #                 zip(
    #                     self.last_cellular_state,
    #                     self.last_cellular_action,
    #                 )
    #             )
    #             if distinct_pair_set <= set(last_pair_list): # check
    #                 for last_pair in last_pair_list: # count
    #                     if last_pair in distinct_pair_set:
    #                         self.transfervk[s, a] += 1


    def estimates(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                maxN = max([1, self.Nk[s, a]])
                self.r_estimate[s, a] = self.Rk[s, a] / maxN
                for next_s in range(self.prior_knowledge.n_states):
                    if self.prior_knowledge.identical_intracellular_transitions is False:
                        self.p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / maxN
                    else:
                        p_estimate = lambda si, ai, next_si: self.intracellularPk[si,ai,next_si] / max(1, self.intracellularNk[si,ai])
                        self.p_estimate[s, a, next_s] = np.prod(
                            [
                                p_estimate(si, ai, next_si) for si, ai, next_si in zip(
                                    tabular2cellular(s, self.prior_knowledge.state_space),
                                    tabular2cellular(a, self.prior_knowledge.action_space),
                                    tabular2cellular(next_s, self.prior_knowledge.state_space),
                                )
                            ]
                        )
                        # maxN = max([1, self.transferNk[s,a]])
                        # self.p_estimate[s, a, next_s] = self.transferPk[s, a, next_s] / maxN
                    assert 0 <= self.r_estimate[s, a] <= 1
                    assert 0 <= self.p_estimate[s, a, next_s] <= 1
    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs)
    def distances(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                maxN = max([1, self.Nk[s, a]])
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.prior_knowledge.n_states * self.prior_knowledge.n_actions * self.t / self.prior_knowledge.confidence_level))
                                                 / (2 * maxN))
                if self.prior_knowledge.identical_intracellular_transitions is True:
                    maxN = max([1, self.transferNk[s, a]])
                self.p_distances[s, a] = np.sqrt((14 * self.prior_knowledge.n_states * np.log(2 * self.prior_knowledge.n_actions * self.t / self.prior_knowledge.confidence_level))
                                                / (maxN))

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.prior_knowledge.n_states)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1: 
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  # Error? 
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=int(1e6)): # max_iter=1000
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
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] if self.transition_indicator[s,a]==1 else -np.inf for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                sampled_cellular_action = tabular2cellular(
                    np.random.choice(choice),
                    self.prior_knowledge.action_space,
                )
                self.policy[:, s] = cp.copy(sampled_cellular_action)

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


    # To start a new episode (init var, computes estmates and run EVI).
    def off_policy(self):
        if self.new_episode and not self.new_pruning:
            self.updateN()
            self.vk = np.zeros(
                shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
                dtype=int,
            )
            if self.prior_knowledge.identical_intracellular_transitions is True:
                if not hasattr(self.prior_knowledge, 'reward_func'):
                    self.update_transferv()
                    assert (self.transfervk >= self.vk).all()
                self.update_intracellularN()
                self.update_transferN()
                assert (self.transferNk >= self.Nk).all()
                self.intracellularvk = np.zeros(
                    shape=(self.prior_knowledge.n_intracellular_states, self.prior_knowledge.n_intracellular_actions),
                    dtype=int,
                )
            self.estimates()
            self.distances()
        behaviour_policy = cp.copy(self.policy)
        self.EVI(self.r_estimate, self.p_estimate, epsilon=1. / max(1, self.t))
        target_policy = cp.copy(self.policy)
        self.pe_shield(behaviour_policy, target_policy, self.p_estimate)



    def stopping_criterion(self):
        if self.prior_knowledge.identical_intracellular_transitions is True and hasattr(self.prior_knowledge, 'reward_func'):
            self.new_episode = self.transfervk[self.last_tabular_state, self.last_tabular_action] >= max([1, self.transferNk[self.last_tabular_state, self.last_tabular_action]])
        else:
            self.new_episode = self.vk[self.last_tabular_state, self.last_tabular_action] >= max([1, self.Nk[self.last_tabular_state, self.last_tabular_action]])


    def sample_action(self, state):
        self.last_cellular_state = self.prior_knowledge.cellularize(
            element=state,
            space=self.prior_knowledge.state_space,
        )
        self.last_tabular_state = cellular2tabular(
            self.last_cellular_state,
            self.prior_knowledge.state_space,
        )
        assert self.last_tabular_state == self.current_tabular_state
        self.last_cellular_action = cp.copy(self.policy[:, self.last_tabular_state])
        self.last_tabular_action = cellular2tabular(
            self.last_cellular_action,
            self.prior_knowledge.action_space,
        )
        self.stopping_criterion()
        self.data['off_policy_time'] = np.nan
        self.data['updated_cells'] = ''
        if self.new_episode or self.new_pruning:
            self.data['off_policy_time'] = time.perf_counter()
            self.off_policy()
            self.last_cellular_action = cp.copy(self.policy[:, self.last_tabular_state])
            self.last_tabular_action = cellular2tabular(
                self.last_cellular_action,
                self.prior_knowledge.action_space,
            )
        self.data['off_policy_time'] = time.perf_counter() - self.data['off_policy_time']
        output = self.prior_knowledge.decellularize(
            cellular_element=self.last_cellular_action,
            space=self.prior_knowledge.action_space,
        )
        return output

    # To update the learner after one step of the current policy.
    def update(self, state, reward, info):
        self.current_cellular_state = self.prior_knowledge.cellularize(
            element=state,
            space=self.prior_knowledge.state_space,
        )
        self.current_tabular_state = cellular2tabular(
            self.current_cellular_state,
            self.prior_knowledge.state_space,
        )
        self.current_reward = reward
        self.side_effects_processing(info['side_effects'])
        self.action_pruning()
        self.updatev()
        self.updateP()
        self.updateR()
        if self.prior_knowledge.identical_intracellular_transitions is True:
            self.update_intracellularP()
            self.update_intracellularv()
            assert np.array([self.intracellularvk[s,a] + self.intracellularNk[s,a] >= max(self.intracellularPk[s,a,:]) for s in range(self.prior_knowledge.n_intracellular_states) for a in range(self.prior_knowledge.n_intracellular_actions)]).all()
            if hasattr(self.prior_knowledge, 'reward_func'):
                self.update_transferv()
                assert (self.transfervk >= self.vk).all()
        self.t += 1

    # Registering new side effects
    def side_effects_processing(self, side_effects):
        for reporting_cell in range(self.prior_knowledge.n_cells):
            for reported_cell in range(self.prior_knowledge.n_cells):
                reported_current_intracellular_state = self.current_cellular_state[reported_cell]
                if side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_funcs[reported_current_intracellular_state] -= {'unsafe'}
                elif side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_funcs[reported_current_intracellular_state] -= {'safe'}


    def reward_shaping(self, tabular_state, tabular_action):
        cellular_state = tabular2cellular(
            tabular_state,
            self.prior_knowledge.state_space,
        )
        for si in cellular_state:
            if self.side_effects_funcs[si] == {'safe', 'unsafe'}:
                return self.r_distances[tabular_state, tabular_action]
        return 0.0


    # Action pruning
    def action_pruning(self):

        # initialization
        self.new_pruning = False
        # basic case
        for cell in range(self.prior_knowledge.n_cells):
            if self.side_effects_funcs[self.current_cellular_state[cell]] == {'unsafe'}:
                if self.intracellular_transition_indicator[self.last_cellular_state[cell], self.last_cellular_action[cell]] == 1:
                    self.new_pruning = True
                self.intracellular_transition_indicator[self.last_cellular_state[cell], self.last_cellular_action[cell]] = 0
        # corner cases
        for cell in range(self.prior_knowledge.n_cells):
            n_unpruned_actions = np.sum(self.intracellular_transition_indicator[self.last_cellular_state[cell], :])
            if n_unpruned_actions >= 2:
                self.path[cell] = set()
            elif n_unpruned_actions == 1:
                self.path[cell].add((self.last_cellular_state[cell], self.last_cellular_action[cell]))
            if self.side_effects_funcs[self.current_cellular_state[cell]] == {'unsafe'} or n_unpruned_actions == 0:
                for (si, ai) in self.path[cell]:
                    if self.intracellular_transition_indicator[si, ai] == 1:
                        self.new_pruning = True
                    self.intracellular_transition_indicator[si, ai] = 0
        # update transition indicators
        if self.new_pruning:
            for s in range(self.prior_knowledge.n_states):
                for a in range(self.prior_knowledge.n_actions):
                    for si, ai in zip(
                            tabular2cellular(s, self.prior_knowledge.state_space),
                            tabular2cellular(a, self.prior_knowledge.action_space)
                        ):
                        if self.intracellular_transition_indicator[si, ai] == 0:
                            self.transition_indicator[s, a] = 0
                            break


    # Applying shielding
    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            reset = self.policy_update[cell] == 1
            reset = (self.last_cellular_state[cell] in self.initial_safe_intracellular_states) and reset
            if reset:
                self.rc[cell] += 1.0
            reset = (np.random.rand() <= 1.0 / max([1.0, self.rc[cell]])) and reset
            if reset:
                tmp_policy[cell, :] = cp.copy(self.initial_policy[cell, :])
                self.policy_update[cell] = 0
                self.data['updated_cells'] = self.data['updated_cells'] + '(' + str(cell) + ')' + '|'
            else:
                tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
                if self.policy_update[cell] == 0:
                    initial_policy_is_updated = True
                    self.policy_update[cell] = 1
                else:
                    initial_policy_is_updated = False
                verified = self.verify_with_prism(tmp_policy, p_estimate)
                if not verified:
                    tmp_policy[cell, :] = cp.copy(behaviour_policy[cell, :])
                    if initial_policy_is_updated:
                        self.policy_update[cell] = 0
                else:
                    self.data['updated_cells'] = self.data['updated_cells'] + str(cell) + '|'
        self.policy = cp.copy(tmp_policy)

    # Prism
    def verify_with_prism(
        self,
        tmp_policy,
        p_estimate,
        n_attempts=3,
    ):
        
        self.data['prism_error'] = ''
        for attempt in range(n_attempts):
            try:
                self.initialize_prism_files()
                self.write_model_file(tmp_policy, p_estimate)
                verified = self.run_prism()
                break
            except PrismError as error:
                if attempt == n_attempts - 1:
                    #raise PrismError
                    verified = False
                    self.data['prism_error'] = str(error)
                os.system('rm -r -f ' + self.prism_path) # clean
        os.system('rm -r -f ' + self.prism_path) # clean
        return verified
    
    def initialize_prism_files(self):
        cpu_id = Process().cpu_num()
        tmp_id = np.random.randint(0, time.time_ns())
        self.prism_path = '.prism_tmps/' + str(cpu_id) + str(tmp_id)[-5:] + '/'
        while True:
            try:
                os.mkdir(self.prism_path)
                break
            except FileExistsError:
                time.sleep(0.1)
        with open(self.prism_path + 'constraints.props', 'w') as props_file:
            props_file.write(self.prism_props)
    
    def write_model_file(
            self,
            tmp_policy,
            p_estimate,
            epsilon: float = 0.000000000000001,
        ):
        
        os.system('rm -fr ' + self.prism_path + 'model.prism')
        with open(self.prism_path + 'model.prism', 'a') as prism_file:

            prism_file.write('dtmc\n\n')

            for s in range(self.prior_knowledge.n_states):
                for cell in range(self.prior_knowledge.n_cells):
                    C = 0
                    if self.policy_update[cell] == 1:
                        state = tabular2cellular(
                            s,
                            self.prior_knowledge.state_space,
                        )
                        for si in state:
                            if 'unsafe' in self.side_effects_funcs[si]:
                                C = 1
                                break
                    prism_file.write('const int C' + str(s) + '_' + str(cell) + ' = ' + str(C) + ';\n')
            prism_file.write('\n')

            prism_file.write('module M\n\n')

            prism_file.write('s : [0..' + str(self.prior_knowledge.n_states) + '] init ' + str(self.last_tabular_state) + ';\n')
            for cell in range(self.prior_knowledge.n_cells):
                prism_file.write('c_' + str(cell) + ' : [0..1] init C' + str(self.last_tabular_state) + '_' + str(cell) + ';\n')
            prism_file.write('\n')

            for s in range(self.prior_knowledge.n_states):
                prism_file.write('[] (s = ' + str(s) + ') -> ')
                a = cellular2tabular(
                    tmp_policy[:, s], 
                    self.prior_knowledge.action_space,
                )
                init_iter = True
                for next_s in range(self.prior_knowledge.n_states):
                    lb = max(
                        epsilon,
                        max(
                            0,
                            p_estimate[s, a, next_s] - self.p_distances[s, a]
                        ),
                    )
                    ub = min(
                        1-epsilon,
                        min(
                            1,
                            p_estimate[s, a, next_s] + self.p_distances[s, a]
                        ),
                    )
                    if not init_iter:
                        prism_file.write(' + ')
                    prism_file.write('[' + str(lb) + ',' + str(ub) + "] : (s' = " + str(next_s) + ')')
                    for cell in range(self.prior_knowledge.n_cells):
                        prism_file.write(' & (c_' + str(cell) + "' = C" + str(next_s) + '_' + str(cell) + ')')
                    init_iter = False
                prism_file.write(';\n')
            prism_file.write('\n')

            prism_file.write('endmodule\n\n')

            prism_file.write("formula n = ")
            for cell in range(self.prior_knowledge.n_cells):
                prism_file.write("c_" + str(cell)+ " + ")
            prism_file.write("0;\n")
            for count, cell_class in enumerate(self.prior_knowledge.cell_classes):
                prism_file.write("formula n_" + cell_class + " = ")
                for cell in self.prior_knowledge.cell_labelling[count]:
                    prism_file.write("c_" + str(cell) + " + ")
                prism_file.write("0;\n")

    def run_prism(self):
        try:
            command = ['prism/prism/bin/prism', self.prism_path + 'model.prism', self.prism_path + 'constraints.props']
            output = subprocess.check_output(command, timeout=None)
            self.prism_output = output # for debugging purposes
        except subprocess.CalledProcessError as error:
            self.prism_output = error.output.decode() # for debugging purposes
            with open('.prism_tmps/error.txt', 'a') as error_file:
                error_file.write(error.output.decode())
            raise PrismError('Prism returned an error. See ".prism_tmps/error.txt" for details.')
        output = output.decode()
        occurances = 0
        for line in output.splitlines():
            if 'Result:' in line:
                occurances += 1
                if 'true' in line:
                    verified = True
                elif 'false' in line:
                    verified = False
                else:
                    raise PrismError('Verification returned non-Boolean result.')
        if occurances != 1:
            raise PrismError('Verification returned ' + str(occurances) + ' results. Expected 1 Boolean result.')
        return verified


    # To get the data to save.
    def get_data(self):
        self.data['name'] = self.name()
        if np.isnan(self.data['off_policy_time']):
            self.data['off_policy_time'] = ''
        if self.new_episode and self.new_pruning:
            self.data['update_kinds'] = 'both'
        elif self.new_episode:
            self.data['update_kinds'] = 'episode'
        elif self.new_pruning:
            self.data['update_kinds'] = 'pruning'
        else:
            self.data['update_kinds'] = ''
            self.data['prism_error'] = ''
        self.data['updated_cells'] = self.data['updated_cells'][:-1]
        self.data['regulatory_constraints'] = self.regulatory_constraints
        return self.data
    
    
class PrismError(Exception):
    pass

