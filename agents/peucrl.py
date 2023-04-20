"""Copypasted and modified, cite"""

import copy as cp
from agents.utils import * #from learners.discreteMDPs.utils import *
from time import perf_counter_ns

#from learners.discreteMDPs.AgentInterface import Agent

# modifications:

from copy import deepcopy
import numpy as np
from os import system
from psutil import Process
import subprocess


class PeUcrlAgent:

    """This script adds action pruning"""

    def __init__(
        self,
        confidence_level: float, # a parameter
        n_cells: int,
        n_intracellular_states: int,
        cellular_encoding, # states to N^n
        n_intracellular_actions: int,
        cellular_decoding, # N^n to actions # the (size, coding) pair is equivalent to the state/action space
        initial_policy: np.ndarray,
        reward_function = None,
        cell_classes = None,
        cell_labelling_function = None,
        regulatory_constraints = None, 
        seed=0,
    ):

        self.n_states = n_intracellular_states ** n_cells
        self.n_actions = n_intracellular_actions ** n_cells
        self.t = 1
        self.delta = confidence_level

        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.n_intracellular_actions = n_intracellular_actions
        self.cellular_encoding = cellular_encoding
        self.cellular_decoding = cellular_decoding
        np.random.seed(seed=seed)

        self.observations = [[], [], []] # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros((self.n_states, self.n_actions)) #the state-action count for the current episode k
        self.Nk = np.zeros((self.n_states, self.n_actions)) #the state-action count prior to episode k

        self.r_distances = np.zeros((self.n_states, self.n_actions))
        self.p_distances = np.zeros((self.n_states, self.n_actions))
        self.Pk = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.Rk = np.zeros((self.n_states, self.n_actions))

        self.u = np.zeros(self.n_states)
        self.span = []
        self.initial_policy = initial_policy
        self.policy = self.initial_policy
        # self.policy = np.zeros((self.n_states, self.n_actions)) # policy
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         if cellular2tabular(initial_policy[:, s], self.n_intracellular_actions, self.n_cells) == a:
        #             self.policy[s, a] = 1.0

        # modifications:

        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints
        self.cell_Nk = np.zeros(shape=[self.n_states, self.n_actions])
        self.policy_update = np.zeros(shape=[self.n_cells], dtype=bool)
        self.side_effects_functions = [{'safe', 'unsafe'} for _ in range(self.n_intracellular_states)]


        # initialise prism
        self.cpu_id = Process().cpu_num()
        self.prism_path = 'agents/prism_files/cpu_' + str(self.cpu_id) + '/'
        system('rm -r -f ' + self.prism_path + '; mkdir ' + self.prism_path)
        with open(self.prism_path + 'constraints.props', 'a') as props_file:
            props_file.write(self.regulatory_constraints)

        self.intracellular_transition_indicators = np.ones((self.n_intracellular_states, self.n_intracellular_actions), dtype=int)
        self.transition_indicators = np.ones((self.n_states, self.n_actions), dtype=int)
        self.new_pruning = False

    def name(self):
        return "PEUCRL"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Nk[s, a] += self.vk[s, a]
                self.cell_Nk[s, a] += self.cell_vk[s, a] # mod

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        for tabular_state in range(self.n_states):
            for intra_state in tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells):
                if intra_state == self.observations[0][-2]:
                    for tabular_action in range(self.n_actions):
                        for intra_action in tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells):
                            if intra_action == self.observations[1][-1]:
                                for tabular_next_state in range(self.n_states):
                                    for intra_next_state in tabular2cellular(tabular_next_state, self.n_intracellular_states, self.n_cells):
                                        if intra_next_state == self.observations[0][-1]:
                                            self.Pk[tabular_state, tabular_action, tabular_next_state] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.n_states * self.n_actions * self.t / self.delta))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.n_states * np.log(2 * self.n_actions * self.t / self.delta))
                                                 / (max([1, self.cell_Nk[s, a]]))) #mod

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.n_states)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1: 
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.n_states)
        sorted_indices = np.arange(self.n_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.n_states):

                temp = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                    temp[a] *= self.transition_indicators[s,a] # mod
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg] # not modified for the sake of rewards
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                action = np.random.choice(choice)
                self.policy[:, s] = tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
                #self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.n_actions)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.n_states)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break


    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.start_episode = perf_counter_ns()
        if self.stopping:
            self.updateN()
            self.vk = np.zeros((self.n_states, self.n_actions))
            self.cell_vk = np.zeros(shape=[self.n_states, self.n_actions])
        r_estimate = np.zeros((self.n_states, self.n_actions))
        p_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                div = max([1, self.cell_Nk[s, a] + self.cell_vk[s, a]]) # mod
                r_estimate[s, a] = self.Rk[s, a] / max([1, self.Nk[s, a] + self.vk[s, a]]) # mod
                for next_s in range(self.n_states):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        behaviour_policy = deepcopy(self.policy)
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))
        target_policy = deepcopy(self.policy)
        self.pe_shield(behaviour_policy, target_policy, p_estimate)
        self.end_episode = perf_counter_ns()

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.n_states, self.n_actions))
        self.Nk = np.zeros((self.n_states, self.n_actions))
        self.cell_vk = np.zeros((self.n_states, self.n_actions)) # mod
        self.cell_Nk = np.zeros((self.n_states, self.n_actions)) # mod
        self.u = np.zeros(self.n_states)
        self.Pk = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.Rk = np.zeros((self.n_states, self.n_actions))
        self.span = [0]
        self.policy = self.initial_policy
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         self.policy[s, a] = 1. / self.n_actions
        # self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def sample_action(self, previous_state):
        s = self.cellular_encoding(previous_state)
        state = cellular2tabular(s, self.n_intracellular_states, self.n_cells)
        if self.t == 1:
            self.reset(state)
        action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
        #action = categorical_sample([self.policy[state, a] for a in range(self.n_actions)], np.random)
        self.previous_state = state # moved
        self.start_episode = np.nan
        self.end_episode = np.nan
        self.stopping = self.vk[state, action] >= max([1, self.Nk[state, action]])
        if self.stopping or self.new_pruning:
            self.new_episode()
            action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
            #action = categorical_sample([self.policy[state, a] for a in range(self.n_actions)], np.random)
        self.previous_action = action
        tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
        action = self.cellular_decoding(action)
        return tabular2cellular(action, self.n_intracellular_actions, self.n_cells)

    # modification:

    def updatev(self):
        self.vk[self.previous_state, self.previous_action] += 1 #standard
        for tabular_state in range(self.n_states):
            for tabular_action in range(self.n_actions):
                for intra_state, intra_action in zip(
                    tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells),
                    tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells),
                ):
                    for previous_intra_state, previous_intra_action in zip(
                        tabular2cellular(self.previous_state, self.n_intracellular_states, self.n_cells),
                        tabular2cellular(self.previous_action, self.n_intracellular_actions, self.n_cells),
                    ):
                        if intra_state == previous_intra_state and intra_action == previous_intra_action:
                            self.cell_vk[tabular_state, tabular_action] += 1

        # for tabular_state in range(self.n_states):
        #     for intra_state in tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells):
        #         if intra_state in tabular2cellular(self.observations[0][-2], self.n_intracellular_states, self.n_cells):
        #             for tabular_action in range(self.n_actions):
        #                 for intra_action in tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells):
        #                     if intra_action in tabular2cellular(self.observations[1][-1], self.n_intracellular_actions, self.n_cells):
        #                         self.cell_vk[tabular_state, tabular_action] += 1


    # To update the learner after one step of the current policy.
    def update(self, current_state, reward, side_effects):
        observation = self.cellular_encoding(current_state)
        observation = cellular2tabular(observation, self.n_intracellular_states, self.n_cells)
        self.observations[0].append(observation)
        self.observations[1].append(self.previous_action)
        self.observations[2].append(reward)
        self.updatev()
        self.updateP()
        self.updateR()
        self.side_effects_processing(side_effects, observation)
        self.action_pruning()
        self.t += 1

    def action_pruning(self):

        self.start_time_step = perf_counter_ns()
        
        new_pruning = False

        current_state = tabular2cellular(self.observations[0][-1], self.n_intracellular_states, self.n_cells)
        action = tabular2cellular(self.observations[1][-1], self.n_intracellular_actions, self.n_cells)
        previous_state = tabular2cellular(self.observations[0][-2], self.n_intracellular_states, self.n_cells)
        # basic case
        for cell in range(self.n_cells):
            if {'unsafe'} == self.side_effects_functions[current_state[cell]]:
                if self.intracellular_transition_indicators[previous_state[cell], action[cell]] == 1:
                    new_pruning = True
                self.intracellular_transition_indicators[previous_state[cell], action[cell]] = 0
        
        # corner cases
        if self.t == 1:
            self.path = [set() for _ in range(self.n_cells)]
        for cell in range(self.n_cells):
            n_unpruned_actions = np.sum(self.intracellular_transition_indicators[previous_state[cell], :])
            if n_unpruned_actions >= 2:
                self.path[cell] = set()
            elif n_unpruned_actions == 1:
                self.path[cell].add((previous_state[cell], action[cell]))
            if ({'unsafe'} == self.side_effects_functions[current_state[cell]]) or n_unpruned_actions == 0:
                for (intracellular_state, intracellular_action) in self.path[cell]:
                    if self.intracellular_transition_indicators[intracellular_state, intracellular_action] == 1:
                        new_pruning = True
                    self.intracellular_transition_indicators[intracellular_state, intracellular_action] = 0

        if new_pruning:
            for flat_state in range(self.n_states):
                for flat_action in range(self.n_actions):
                    for (intracellular_state, intracellular_action) in zip(
                                tabular2cellular(flat_state, self.n_intracellular_states, self.n_cells),
                                tabular2cellular(flat_action, self.n_intracellular_actions, self.n_cells)
                        ):
                        if self.intracellular_transition_indicators[intracellular_state, intracellular_action] == 0:
                            self.transition_indicators[flat_state, flat_action] = 0
                            break
        
        self.new_pruning = new_pruning

        self.end_time_step = perf_counter_ns()

    def side_effects_processing(self, side_effects, observation):

        current_state = tabular2cellular(observation, self.n_intracellular_states, self.n_cells)
        for reporting_cell in range(self.n_cells):
            for reported_cell in range(self.n_cells):
                if side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_functions[current_state[reported_cell]] -= {'unsafe'}
                elif side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_functions[current_state[reported_cell]] -= {'safe'}

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        
        tmp_policy = deepcopy(behaviour_policy)
        self.previous_policy = deepcopy(behaviour_policy) # for evaluations
        cell_set = set(range(self.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            tmp_policy[cell, :] = target_policy[cell, :]
            if self.policy_update[cell] == 0:
                initial_policy_is_updated = True
                self.policy_update[cell] = 1
            else:
                initial_policy_is_updated = False
            verified = self.verify_with_prism(tmp_policy, p_estimate)
            if not verified:
                tmp_policy[cell, :] = deepcopy(behaviour_policy[cell, :])
                if initial_policy_is_updated:
                    self.policy_update[cell] = 0
        self.policy = deepcopy(tmp_policy)

    # Prism
    def verify_with_prism(
        self,
        tmp_policy,
        p_estimate,
    ):
        
        self.write_model_file(tmp_policy, p_estimate)
        try:
            output = subprocess.check_output(['prism/prism/bin/prism', self.prism_path + 'model.prism', self.prism_path + 'constraints.props'])
        except subprocess.CalledProcessError as error:
            error = error.output
            error = error.decode()
            print(error)
            raise ValueError('Prism returned an error, see above.')
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
                    raise ValueError('Verification returned non-Boolean result.')
        if occurances != 1:
            raise ValueError('Verification returned ' + str(occurances) + ' results. Expected 1 Boolean result.')
        self.prism_output = output # for debugging purposes

        return verified
    
    def write_model_file(
            self,
            tmp_policy,
            p_estimate,
            epsilon: float = 0.000000000000001,
        ):
        
        system('rm -fr ' + self.prism_path + 'model.prism')
        with open(self.prism_path + 'model.prism', 'a') as prism_file:

            prism_file.write('dtmc\n\n')

            for flat_state in range(self.n_states):
                for cell in range(self.n_cells):
                    C = 0
                    if self.policy_update[cell] == 1:
                        intracellular_states_set = tabular2cellular(flat_state, self.n_intracellular_states, self.n_cells)
                        for intracellular_state in intracellular_states_set:
                            if 'unsafe' in self.side_effects_functions[intracellular_state]:
                                C = 1
                                break
                    prism_file.write('const int C' + str(flat_state) + '_' + str(cell) + ' = ' + str(C) + ';\n')
            prism_file.write('\n')

            prism_file.write('module M\n\n')

            prism_file.write('s : [0..' + str(self.n_states) + '] init ' + str(self.previous_state) + ';\n')
            for cell in range(self.n_cells):
                prism_file.write('c_' + str(cell) + ' : [0..1] init C' + str(self.previous_state) + '_' + str(cell) + ';\n')
            prism_file.write('\n')

            for flat_state in range(self.n_states):
                prism_file.write('[] (s = ' + str(flat_state) + ') -> ')
                flat_action = cellular2tabular(tmp_policy[:, flat_state], self.n_intracellular_actions, self.n_cells)
                init_iter = True
                for next_flat_state in range(self.n_states):
                    lb = max(
                        [epsilon,
                         p_estimate[flat_state, flat_action, next_flat_state] - self.p_distances[flat_state, flat_action]]
                    )
                    ub = min(
                        [1-epsilon,
                         p_estimate[flat_state, flat_action, next_flat_state] + self.p_distances[flat_state, flat_action]]
                    )
                    if not init_iter:
                        prism_file.write(' + ')
                    prism_file.write('[' + str(lb) + ',' + str(ub) + "] : (s' = " + str(next_flat_state) + ')')
                    for cell in range(self.n_cells):
                        prism_file.write(' & (c_' + str(cell) + "' = C" + str(next_flat_state) + '_' + str(cell) + ')')
                    init_iter = False
                prism_file.write(';\n')
            prism_file.write('\n')

            prism_file.write('endmodule\n\n')

            prism_file.write("formula n = ")
            for cell in range(self.n_cells):
                prism_file.write("c_" + str(cell)+ " + ")
            prism_file.write("0;\n")
            for count, cell_class in enumerate(self.cell_classes):
                prism_file.write("formula n_" + cell_class + " = ")
                for cell in self.cell_labelling_function[count]:
                    prism_file.write("c_" + str(cell) + " + ")
                prism_file.write("0;\n")

    # subroutines the user can call to collect data

    def get_ns_between_time_steps(self):
        # this has been changed to measure the length of action pruning
        return self.end_time_step - self.start_time_step
    
    def get_ns_between_episodes(self):
        return self.end_episode - self.start_episode
    
    def get_update_type(self):

        output = {'new_episode': [], 'new_action_pruning': []}
        def loop(policy, previous_policy, n_cells):
            output = []
            for cell in range(n_cells):
                if (policy[cell,:] != previous_policy[cell,:]).any():
                    output.append(cell)
            return output
        if self.stopping and self.new_pruning:
            output['new_episode'] = loop(self.policy, self.previous_policy, self.n_cells)
            output['new_action_pruning'] = loop(self.policy, self.previous_policy, self.n_cells)
        elif self.stopping:
            output['new_episode'] = loop(self.policy, self.previous_policy, self.n_cells)
        elif self.new_pruning:
            output['new_action_pruning'] = loop(self.policy, self.previous_policy, self.n_cells)
        return deepcopy(output) 
            

