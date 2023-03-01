#from prism import verify
import numpy as np
import random
from itertools import chain
from time import perf_counter_ns
import math
from agents.utils.space_transformations import cellular2tabular, tabular2cellular
from os import system
from copy import deepcopy, copy
from pprint import pprint



class PeUcrlAgent:

    def __init__(
        self,
        confidence_level: float, # a parameter
        accuracy: float, # a parameter
        n_cells: int, # prior knowledge from here and down
        n_intracellular_states: int,
        cellular_encoding, # states to N^n
        n_intracellular_actions: int,
        cellular_decoding, # N^n to actions # the (size, coding) pair is equivalent to the state/action space
        cell_classes: set,
        cell_labelling_function,
        regulatory_constraints,
        initial_policy: np.ndarray, # should be in a cellular encoding
        reward_function, # todo: make this optional
    ):

        # check correctness of inputs
        assert 0 < confidence_level < 1
        assert 0 < accuracy

        # point inputs to self
        self.confidence_level = confidence_level
        self.accuracy = accuracy
        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.cellular_encoding = cellular_encoding
        self.n_intracellular_actions = n_intracellular_actions
        self.cellular_decoding = cellular_decoding
        #self.reward_function = lambda flat_state, flat_action: reward_function(self._unflatten(flat_state=flat_state), self._unflatten(flat_action=flat_action))
        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints

        # compute additional parameters
        self.n_states = n_intracellular_states ** n_cells
        self.n_actions = n_intracellular_actions ** n_cells

        self.reward_function = np.zeros((self.n_states, self.n_actions))
        for flat_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                self.reward_function[flat_state,flat_action] = reward_function(flat_state,flat_action)

        # initialise behaviour policy
        self.initial_policy = initial_policy
        self.behaviour_policy = deepcopy(self.initial_policy) # using some kind of conversion?
        self.target_policy = deepcopy(self.initial_policy) #np.zeros((self.n_cells, self.n_states), dtype=int)
        self.policy_update = np.zeros(self.n_cells, dtype=int)

        # initialise counts to 0
        self.time_step = 0
        #self.current_episode_time_step = 0 # do I need this?
        self.current_episode_count = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.previous_episodes_count = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.cellular_current_episode_count = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.cellular_previous_episodes_count = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.intracellular_episode_count = np.zeros((self.n_intracellular_states, self.n_intracellular_actions), dtype=int)
        self.intracellular_sum = np.zeros((self.n_intracellular_states, self.n_intracellular_actions), dtype=int)
        self.intracellular_transition_sum = np.zeros((self.n_intracellular_states, self.n_intracellular_actions, self.n_intracellular_states), dtype=int)

        # initialise statistics
        self.side_effects_functions = [{'safe', 'unsafe'} for _ in range(n_intracellular_states)]
        self.intracellular_transition_estimates = np.zeros((self.n_intracellular_states, self.n_intracellular_actions, self.n_intracellular_states))
        self.intracellular_transition_errors = np.zeros((self.n_intracellular_states, self.n_intracellular_actions))
        self.intracellular_transition_indicators = np.ones((self.n_intracellular_states, self.n_intracellular_actions))
        self.transition_estimates = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.transition_errors = np.zeros((self.n_states, self.n_actions))

        # miscellaneous initialisations
        self.previous_state = None
        self.current_state = None
        self.action_sampled = False

        self.n_policy_changes = 0

        # initialise prism
        system("cd agents/prism; rm -f constraints.props; touch constraints.props")
        props_file = open('agents/prism/constraints.props', 'a')
        props_file.write(self.regulatory_constraints)
        props_file.close()
        #system("rm -f agents/prism/constraints.props")
        #system("cp " + self.regulatory_constraints + " agents/prism/constraints.props")

        system('rm -f agents/prism/model.prism')
        system("touch agents/prism/model.prism")
        prism_file = open('agents/prism/model.prism', 'a')

        prism_file.write('dtmc\n\n')

        for flat_state in range(self.n_intracellular_states):
            prism_file.write('const int C_' + str(flat_state) + ';\n')
        prism_file.write('\n')
        
        for cell in range(self.n_cells):

            prism_file.write('const int sinit' + str(cell) + ';\n')
            prism_file.write('const int cinit' + str(cell) + ';\n')
            prism_file.write('const int piupdate' + str(cell) + ';\n\n')

            for flat_state in range(self.n_intracellular_states):
                for flat_next_state in range(self.n_intracellular_states-1):
                    prism_file.write('const double p' + str(cell) + "_" + str(flat_state) + "_" + str(flat_next_state) + ';\n')
                prism_file.write('const double p' + str(cell) + "_" + str(flat_state) + "_" + str(self.n_intracellular_states-1) + ' = 1')
                for flat_next_state in range(self.n_intracellular_states-1):
                    prism_file.write(' - p' + str(cell) + "_" + str(flat_state) + "_" + str(flat_next_state))
                prism_file.write(';\n\n')
            
            prism_file.write('module cell' + str(cell) + '\n\n')

            prism_file.write('s' + str(cell) + ' : [0..' + str(self.n_intracellular_states) + '] init sinit' + str(cell) + ';\n')
            prism_file.write('c' + str(cell) + ' : [0..1] init cinit' + str(cell) + ';\n\n')

            for flat_state in range(self.n_intracellular_states):
                prism_file.write("[] s" + str(cell) + "=" + str(flat_state) + " -> ")
                for flat_next_state in range(self.n_intracellular_states - 1):
                    prism_file.write('p' + str(cell) + "_" + str(flat_state) + "_" + str(flat_next_state) + ":(s" + str(cell) + "'=" + str(flat_next_state) + ") & (c" + str(cell) + "'=(piupdate" + str(cell) + "*C_" + str(flat_next_state) + ")) + ")
                prism_file.write('p' + str(cell) + "_" + str(flat_state) + "_" + str(self.n_intracellular_states - 1) + ":(s" + str(cell) + "'=" + str(self.n_intracellular_states - 1) + ") & (c" + str(cell) + "'=(piupdate" + str(cell) + "*C_" + str(self.n_intracellular_states - 1) + "));\n")
            
            prism_file.write("\nendmodule\n\n")

        prism_file.write("formula n = ")
        for cell in range(self.n_cells):
            prism_file.write("c" + str(cell) + " + ")
        prism_file.write("0;\n")
        for count, cell_class in enumerate(self.cell_classes):
            prism_file.write("formula n_" + cell_class + " = ")
            for cell in self.cell_labelling_function[count]:
                prism_file.write("c" + str(cell) + " + ")
            prism_file.write("0;\n")

        prism_file.close()


    def sample_action(
        self,
        previous_state,
    ):

        """Sample an action from the behaviour policy and assign current state to previous state"""

        # assert that we put in the right state
        self.action_sampled = True

        # update state, action
        self.previous_state = self.cellular_encoding(previous_state) # dict
        self.flat_previous_state = self._flatten(state=self.previous_state)
        self.cellular_action = deepcopy(self.behaviour_policy[:, self.flat_previous_state])
        self.action = self.cellular_decoding(self.cellular_action)
        return self.action


    def update(
        self,
        current_state,
        reward,
        side_effects=None,
    ):

        """Update the agent's policy and statistics"""

        assert self.action_sampled # avoid double updating

        self.start_time_step = perf_counter_ns()

        self.action_sampled = False
        self.current_state = self.cellular_encoding(deepcopy(current_state)) # dict
        self.flat_action = self._flatten(action=self.action)
        self.flat_current_state = self._flatten(state=self.current_state)
        self.reward = reward
        if side_effects is None:
            print("Warning: No side effects providing, assuming silence")
            side_effects = np.array([['silent' for _ in range(self.n_cells)] for _ in range(self.n_cells)])
        self.side_effects = side_effects

        # on-policy
        self._side_effects_processing()
        self._action_pruning()
        self._update_current_episode_counts() # moved below. Correct?
        self.time_step += 1

        # off-policy
        next_action = deepcopy(self.behaviour_policy[:, self._flatten(state=self.current_state)])
        new_episode = (self.current_episode_count[self._flatten(state=self.current_state), self._flatten(action=next_action)] >= max([1, self.previous_episodes_count[self._flatten(state=self.current_state), self._flatten(action=next_action)]]))
        
        self.end_time_step = perf_counter_ns()
        self.start_episode = np.nan
        self.end_episode = np.nan
        
        if new_episode:

            self.start_episode = perf_counter_ns()

            self._update_confidence_sets()
            self._extended_value_iteration()
            #
            # random exploration
            #self.target_policy = deepcopy(np.random.randint(0, self.n_intracellular_actions, size=(self.n_cells, self.n_states)))
            #
            self._pe_shield()
            self.current_episode_time_step = self.time_step # is this right? If not, could I just use the time step?
            self.previous_episodes_count += self.current_episode_count
            self.cellular_previous_episodes_count += self.cellular_current_episode_count

            self.end_episode = perf_counter_ns()

    # subroutines

    def get_ns_between_time_steps(self):
        return self.end_time_step - self.start_time_step
    
    def get_ns_between_episodes(self):
        return self.end_episode - self.start_episode

    def _update_current_episode_counts(self):

        # intracellular
        for (intracellular_state, intracellular_action) in zip(self.previous_state, self.action):
            self.intracellular_episode_count[intracellular_state, intracellular_action] += 1

        # cellular
        for flat_previous_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                self.cellular_current_episode_count[flat_previous_state, flat_action] = np.amin(
                    [
                        [
                            self.intracellular_episode_count[intracellular_state, intracellular_action] for intracellular_state in self._unflatten(flat_state=flat_previous_state)
                        ] for intracellular_action in self._unflatten(flat_action=flat_action)
                    ]
                )

        # standard
        self.current_episode_count[self.flat_previous_state, self.flat_action] += 1
        

    def _side_effects_processing(self):

        for reporting_cell in range(self.n_cells):
            for reported_cell in range(self.n_cells):
                if self.side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_functions[self.current_state[reported_cell]] -= {'unsafe'}
                elif self.side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_functions[self.current_state[reported_cell]] -= {'safe'}


    def _action_pruning(self):

        if self.time_step == 0:
            self.n_unpruned_actions = np.zeros(self.n_intracellular_states) + self.n_intracellular_actions

        # basic case
        for cell in range(self.n_cells):
            if 'unsafe' in self.side_effects_functions[self.current_state[cell]]:
                self.intracellular_transition_indicators[self.previous_state[cell], self.action[cell]] = 0
                self.n_unpruned_actions[self.previous_state[cell]] -= 1
        
        # corner cases
        if self.time_step == 0:
            self.path = [set() for _ in range(self.n_cells)]
        for cell in range(self.n_cells):
            if self.n_unpruned_actions[self.previous_state[cell]] >= 2:
                self.path[cell] = set()
            elif self.n_unpruned_actions[self.previous_state[cell]] == 1:
                self.path[cell].add((self.previous_state[cell], self.action[cell]))
            if ('unsafe' in self.side_effects_functions[self.current_state[cell]]) or self.n_unpruned_actions[self.previous_state[cell]] == 0:
                for (intracellular_state, intracellular_action) in self.path[cell]:
                    self.intracellular_transition_indicators[intracellular_state, intracellular_action] = 0
                    self.n_unpruned_actions[intracellular_state] -= 1


    def _update_confidence_sets(self):

        # update transition count
        for cell in range(self.n_cells):
            self.intracellular_sum[self.previous_state[cell], self.action[cell]] += 1
            self.intracellular_transition_sum[self.previous_state[cell], self.action[cell], self.current_state[cell]] += 1

        # update estimates
        for cell in range(self.n_cells):
            self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], self.current_state[cell]] = self.intracellular_transition_sum[self.previous_state[cell], self.action[cell], self.current_state[cell]] / max([1, self.intracellular_sum[self.previous_state[cell], self.action[cell]]])
            # prune
            self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], :]  *= self.intracellular_transition_indicators[self.previous_state[cell], self.action[cell]]
        
        flat_previous_state = self._flatten(state=self.previous_state)
        flat_action = self._flatten(action=self.action)
        flat_current_state = self._flatten(state=self.current_state)
        self.transition_estimates[flat_previous_state, flat_action, flat_current_state] = np.prod(
            [self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], self.current_state[cell]] for cell in range(self.n_cells)]
        )

        # update errors for state--action pairs
        for flat_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                self.transition_errors[flat_state, flat_action] = np.sqrt(
                    (
                        14 * self.n_states * np.log(2 * self.n_actions * self.time_step / self.confidence_level)
                    ) / (
                        max([1, self.cellular_previous_episodes_count[flat_state, flat_action]])
                    )
                )

        # update errors for intracellular state--action pairs
        # note that this does increase the computational complexity, but maybe it is small in comparison to model-checking?
        for intracellular_state in range(self.n_intracellular_states):
            for intracellular_action in range(self.n_intracellular_actions):
                self.intracellular_transition_errors[intracellular_state, intracellular_action] = np.sqrt(
                    (
                        14 * self.n_intracellular_states * np.log(2 * self.n_intracellular_actions * self.time_step / self.confidence_level)
                    ) / (
                        max([1, self.intracellular_episode_count[intracellular_state, intracellular_action]]) # double-check this?
                    )
                )


    def _flatten(
        self,
        state=None,
        action=None,
    ):

        assert (state is None) ^ (action is None)
        if type(state) is np.ndarray:
            m = self.n_intracellular_states
            obj = state
        elif type(action) is np.ndarray:
            m = self.n_intracellular_actions
            obj = action
        else:
            raise ValueError('state or action must be an array')

        return cellular2tabular(obj, m, self.n_cells)
        
    def _unflatten(
        self,
        flat_state=None,
        flat_action=None,
    ):

        assert (flat_state is None) ^ (flat_action is None)
        if type(flat_state) is int:
            m = self.n_intracellular_states
            obj = flat_state
        elif type(flat_action) is int:
            m = self.n_intracellular_actions
            obj = flat_action
        else:
            raise ValueError('flat_state or flat_action must be an integer')

        return tabular2cellular(obj, m, self.n_cells)


    def _inner_max(
        self,
        flat_state,
        flat_action,
        value,
    ):

        #flat_state = self._flatten(state=state)
        initial_sorted_states = list(np.argsort(value))
        max_set = list(np.argwhere(value == np.amax(value))[:,0])
        permuted_max_set = np.random.permutation(max_set)
        sorted_states = [*initial_sorted_states[:-len(max_set)], *permuted_max_set]
        #flat_action = self._flatten(action=action)
        #sorted_states = np.argsort(value)
        max_p = np.zeros(self.n_states)
        for flat_next_state in range(self.n_states):
            max_p[flat_next_state] = self.transition_estimates[flat_state,flat_action,flat_next_state]
        #x = self.transition_estimates[flat_state,flat_action,sorted_states[-1]] + self.transition_errors[flat_state, flat_action]
        #max_p[sorted_states[-1]] = min(
        max_p[sorted_states[-1]] = min(
            [
                1,
                self.transition_estimates[flat_state,flat_action,sorted_states[-1]] + self.transition_errors[flat_state, flat_action]
            ]
        )
        l = -2
        while sum(max_p) > 1:
            max_p[sorted_states[l]] = max(
                [
                    0,
                    1 - sum([max_p[k] for k in chain(range(0, sorted_states[l]), range(sorted_states[l] + 1, self.n_states))])
                ]
            )
            l -= 1
        return sum([ v * p for (v, p) in zip(value, max_p)])   


    def _extended_value_iteration(self):

        previous_value = np.zeros(self.n_states)
        current_value = np.zeros(self.n_states)
        stop = False
        while not stop:
            for flat_state in range(self.n_states):
                values = [(self.reward_function[flat_state, flat_action] + self._inner_max(flat_state, flat_action, previous_value)) for flat_action in range(self.n_actions)]
                current_value[flat_state] = max(values)
                max_flat_action_set = list(np.argwhere(values == np.amax(values))[:,0])
                if len(max_flat_action_set) > 1:
                    print("eq acts:", len(max_flat_action_set))
                max_flat_action = int(random.sample(max_flat_action_set, 1)[0])
                max_action = self._unflatten(flat_action=max_flat_action)
                self.target_policy[:, flat_state] = deepcopy(max_action)
            max_diff = max([current_value[flat_state] - previous_value[flat_state] for flat_state in range(self.n_states)])
            min_diff = min([current_value[flat_state] - previous_value[flat_state] for flat_state in range(self.n_states)])
            stop = (max_diff - min_diff < self.accuracy)
            previous_value = current_value
        self.value_function = current_value # for testing purposes

    def _pe_shield(self):
        
        tmp_policy = deepcopy(self.behaviour_policy)
        cell_set = set(range(self.n_cells))
        while len(cell_set) >= 1:
            cell = self._cell_prioritisation(cell_set)
            cell_set -= {cell}
            tmp_policy[cell, :] = deepcopy(self.target_policy[cell, :])
            verified = self._verify(tmp_policy)
            if verified:
                self.policy_update[cell] == 1
            else:
                tmp_policy[cell, :] = deepcopy(self.behaviour_policy[cell, :])
        if (tmp_policy != self.behaviour_policy).any():
            self.n_policy_changes += 1
        self.behaviour_policy = deepcopy(tmp_policy)
        

    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        cell = random.sample(cell_set, 1)[0]
        return cell

    def _verify(
        self,
        tmp_policy,
    ):

        #verified = True
        verified = self.prism_verify(tmp_policy)

        return verified


    def prism_verify(
        self,
        tmp_policy,
    ):

        # write constants argument
        const_arg = ''
        for intracellular_state in range(self.n_intracellular_states):
            unsafety = int('unsafe' in self.side_effects_functions[intracellular_state])
            const_arg += 'C_' + str(intracellular_state) + '=' + str(unsafety) + ','
        for cell in range(self.n_cells):
            const_arg += 'sinit' + str(cell) + '=' + str(self.current_state[cell]) + ','
            unsafety = int('unsafe' in self.side_effects_functions[self.current_state[cell]])
            const_arg += 'cinit' + str(cell) + '=' + str(unsafety*self.policy_update[cell]) + ','
            const_arg += 'piupdate' + str(cell) + '=' + str(self.policy_update[cell]) + ','
        system('rm -f agents/prism/const.txt')
        system('touch agents/prism/const.txt')
        const_file = open('agents/prism/const.txt', 'w')
        const_file.write(const_arg[:-1])
        const_file.close()

        # write parameters argument, note that the current version of prism can only import this via commandline
        param_arg = ''
        for cell in range(self.n_cells):
            for flat_state in range(self.n_states):
                intracellular_state = self._unflatten(flat_state=flat_state)[cell]
                intracellular_action = tmp_policy[cell, flat_state]
                adds_up = 0 # check
                for next_intracellular_state in range(self.n_intracellular_states):
                    lb = max(
                        [0,
                        self.intracellular_transition_estimates[intracellular_state, intracellular_action, next_intracellular_state] - self.intracellular_transition_errors[intracellular_state, intracellular_action]]
                    )
                    ub = min(
                        [1,
                        self.intracellular_transition_estimates[intracellular_state, intracellular_action, next_intracellular_state] + self.intracellular_transition_errors[intracellular_state, intracellular_action]]
                    )
                    param_arg += 'p' + str(cell) + '_' + str(intracellular_state) + '_' + str(next_intracellular_state) + '=' + str(lb) + ':' + str(ub) + ','
                    adds_up += ub # check
                assert adds_up >= 1
        system('rm -f agents/prism/param.txt')
        system('touch agents/prism/param.txt')
        param_file = open('agents/prism/param.txt', 'w')
        param_file.write(param_arg[:-1])
        param_file.close()
        
        # perform verification
        system('rm -f agents/prism/output.txt')
        system('touch agents/prism/output.txt')
        #system('cd agents/prism; co=$(< const.txt); pa=$(< param.txt); prism model.prism constraints.props -const $co -param $pa &>> output.txt 2>&1')
        system('cd agents/prism; prism model.prism constraints.props -const ' + const_arg[:-1] + ' -param ' + param_arg[:-1] + ' >> output.txt 2>&1')
        output_file = open('agents/prism/output.txt', 'r')
        line_set = output_file.readlines()
        occurances = 0
        for line in reversed(line_set):
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
        output_file.close()

        return verified




