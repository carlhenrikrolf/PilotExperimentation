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

        # assign parameters to self
        self.delta = confidence_level
        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.n_intracellular_actions = n_intracellular_actions
        self.cellular_encoding = cellular_encoding
        self.cellular_decoding = cellular_decoding
        self.initial_policy = initial_policy
        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints

        # calculate additional parameters
        self.n_states = n_intracellular_states ** n_cells
        self.n_actions = n_intracellular_actions ** n_cells

        # initialize counters
        self.t = 1
        self.vk = np.zeros(shape=[self.n_states, self.n_actions], dtype=int) #the state-action count for the current episode k
        self.Nk = np.zeros(shape=[self.n_states, self.n_actions], dtype=int) #the state-action count prior to episode k
        self.Pk = np.zeros(shape=[self.n_states, self.n_actions, self.n_states], dtype=int)
        self.Rk = np.zeros(shape=[self.n_states, self.n_actions], dtype=float)

        # initialize flags
        self.intracellular_transition_indicators = np.ones(shape=[self.n_intracellular_states, self.n_intracellular_actions], dtype=int)
        self.transition_indicators = np.ones(shape=[self.n_states, self.n_actions], dtype=int)
        self.new_pruning = False
        self.policy_update = np.zeros(shape=[self.n_cells], dtype=int)

        # misc initializations
        np.random.seed(seed=seed)
        self.r_distances = np.zeros(shape=[self.n_states, self.n_actions], dtype=float)
        self.p_distances = np.zeros(shape=[self.n_states, self.n_actions], dtype=float)
        self.u = np.zeros(shape=self.n_states, dtype=float)
        self.policy = deepcopy(self.initial_policy)
        self.side_effects_functions = [{'safe', 'unsafe'} for _ in range(self.n_intracellular_states)]
        self.current_state = None

        # # initialise prism
        # self.cpu_id = Process().cpu_num()
        # self.prism_path = 'agents/prism_files/cpu_' + str(self.cpu_id) + '/'
        # system('rm -r -f ' + self.prism_path + '; mkdir ' + self.prism_path)
        # with open(self.prism_path + 'constraints.props', 'a') as props_file:
        #     props_file.write(self.regulatory_constraints)

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Nk[s, a] += self.vk[s, a]

    # Auxiliary function to update v the within-episode state-action count.
    def updatev(self):
        self.vk[self.previous_state, self.previous_action] += 1

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.previous_state, self.previous_action] += self.current_reward

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.previous_state, self.previous_action, self.current_state] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.n_states * self.n_actions * self.t / self.delta))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.n_states * np.log(2 * self.n_actions * self.t / self.delta))
                                                 / (max([1, self.Nk[s, a]])))

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
        u1 = np.zeros(shape=self.n_states, dtype=float)
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
                    temp[a] *= self.transition_indicators[s,a] # modification for action pruning
                    # if no pruning then temp[a] *= 1
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg] # not modified for the sake of rewards
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                action = np.random.choice(choice)
                self.policy[:, s] = tabular2cellular(action, self.n_intracellular_actions, self.n_cells) # modication for cellular
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
    def off_policy(self):

        self.start_episode = perf_counter_ns() # for evaluation

        # the procedure
        if self.new_episode:
            self.updateN()
            self.vk = np.zeros(shape=[self.n_states, self.n_actions], dtype=int)
        r_estimate = np.zeros(shape=[self.n_states, self.n_actions], dtype=float)
        p_estimate = np.zeros(shape=[self.n_states, self.n_actions, self.n_states], dtype=float)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                div = max([1, self.Nk[s, a] + self.vk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.n_states):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        behaviour_policy = deepcopy(self.policy)
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))
        target_policy = deepcopy(self.policy)
        self.pe_shield(behaviour_policy, target_policy, p_estimate)

        self.end_episode = perf_counter_ns() # for evaluation

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def sample_action(self, state):

        self.start_episode = np.nan # for evaluation
        self.end_episode = np.nan # for evaluation

        # the procedure
        state = self.cellular_encoding(state)
        state = cellular2tabular(state, self.n_intracellular_states, self.n_cells)
        self.previous_state = deepcopy(state)
        assert self.previous_state == self.current_state or self.current_state is None
        action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
        self.new_episode = self.vk[state, action] >= max([1, self.Nk[state, action]])
        if self.new_episode or self.new_pruning:
            self.off_policy()
            action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
        self.previous_action = action
        action = tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
        action = self.cellular_decoding(action)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, reward, side_effects):

        state = self.cellular_encoding(state)
        state = cellular2tabular(state, self.n_intracellular_states, self.n_cells)
        self.current_state = deepcopy(state)
        self.current_reward = deepcopy(reward)
        self.updatev()
        self.updateP()
        self.updateR()
        self.side_effects_processing(side_effects, state)
        self.action_pruning()
        self.t += 1

    def action_pruning(self):

        self.start_time_step = perf_counter_ns() # for evaluation
        
        # the procedure

        # initialization
        new_pruning = False
        previous_state = tabular2cellular(self.previous_state, self.n_intracellular_states, self.n_cells)
        previous_action = tabular2cellular(self.previous_action, self.n_intracellular_actions, self.n_cells)
        current_state = tabular2cellular(self.current_state, self.n_intracellular_states, self.n_cells)
        # basic case
        for cell in range(self.n_cells):
            if self.side_effects_functions[current_state[cell]] == {'unsafe'}:
                # maybe move to off-policy (below)
                if self.intracellular_transition_indicators[previous_state[cell], previous_action[cell]] == 1:
                    new_pruning = True
                self.intracellular_transition_indicators[previous_state[cell], previous_action[cell]] = 0
        # corner cases
        if self.t == 1:
            self.path = [set() for _ in range(self.n_cells)]
        for cell in range(self.n_cells):
            n_unpruned_actions = np.sum(self.intracellular_transition_indicators[previous_state[cell], :])
            if n_unpruned_actions >= 2:
                self.path[cell] = set()
            elif n_unpruned_actions == 1:
                self.path[cell].add((previous_state[cell], previous_action[cell]))
            if self.side_effects_functions[current_state[cell]] == {'unsafe'} or n_unpruned_actions == 0:
                # maybe move to off-policy (below)
                for (si, ai) in self.path[cell]:
                    if self.intracellular_transition_indicators[si, ai] == 1:
                        new_pruning = True
                    self.intracellular_transition_indicators[si, ai] = 0
        # maybe move to off-policy (below)
        # update transition indicators
        if new_pruning:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    for si, ai in zip(
                            tabular2cellular(s, self.n_intracellular_states, self.n_cells),
                            tabular2cellular(a, self.n_intracellular_actions, self.n_cells)
                        ):
                        if self.intracellular_transition_indicators[si, ai] == 0:
                            self.transition_indicators[s, a] = 0
                            break
        self.new_pruning = new_pruning

        self.end_time_step = perf_counter_ns() # for evaluation

    def side_effects_processing(self, side_effects, state):

        state = tabular2cellular(state, self.n_intracellular_states, self.n_cells)
        for reporting_cell in range(self.n_cells):
            for reported_cell in range(self.n_cells):
                if side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_functions[state[reported_cell]] -= {'unsafe'}
                elif side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_functions[state[reported_cell]] -= {'safe'}

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        
        tmp_policy = deepcopy(behaviour_policy)
        cell_set = set(range(self.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            tmp_policy[cell, :] = deepcopy(target_policy[cell, :])
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
        
        # initialise prism
        cpu_id = Process().cpu_num()
        random = np.random.randint(0, 1000000)
        self.prism_path = 'agents/prism_files/cpu_' + str(cpu_id) + '_id_' + str(random) + '/'
        system('mkdir ' + self.prism_path)
        with open(self.prism_path + 'constraints.props', 'a') as props_file:
            props_file.write(self.regulatory_constraints)

        # write model file
        self.write_model_file(tmp_policy, p_estimate)

        # verify
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

        # clean
        system('rm -r -f ' + self.prism_path)

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

            for s in range(self.n_states):
                for cell in range(self.n_cells):
                    C = 0
                    if self.policy_update[cell] == 1:
                        state = tabular2cellular(s, self.n_intracellular_states, self.n_cells)
                        for si in state:
                            if 'unsafe' in self.side_effects_functions[si]:
                                C = 1
                                break
                    prism_file.write('const int C' + str(s) + '_' + str(cell) + ' = ' + str(C) + ';\n')
            prism_file.write('\n')

            prism_file.write('module M\n\n')

            prism_file.write('s : [0..' + str(self.n_states) + '] init ' + str(self.previous_state) + ';\n')
            for cell in range(self.n_cells):
                prism_file.write('c_' + str(cell) + ' : [0..1] init C' + str(self.previous_state) + '_' + str(cell) + ';\n')
            prism_file.write('\n')

            for s in range(self.n_states):
                prism_file.write('[] (s = ' + str(s) + ') -> ')
                a = cellular2tabular(tmp_policy[:, s], self.n_intracellular_actions, self.n_cells)
                init_iter = True
                for next_s in range(self.n_states):
                    lb = max(
                        [epsilon,
                         p_estimate[s, a, next_s] - self.p_distances[s, a]]
                    )
                    ub = min(
                        [1-epsilon,
                         p_estimate[s, a, next_s] + self.p_distances[s, a]]
                    )
                    if not init_iter:
                        prism_file.write(' + ')
                    prism_file.write('[' + str(lb) + ',' + str(ub) + "] : (s' = " + str(next_s) + ')')
                    for cell in range(self.n_cells):
                        prism_file.write(' & (c_' + str(cell) + "' = C" + str(next_s) + '_' + str(cell) + ')')
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
            

