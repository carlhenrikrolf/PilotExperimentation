#from prism import verify
import numpy as np
import random
from itertools import chain



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
        reward_function, # todo: make this optional
        cell_classes: set,
        cell_labelling_function,
        regulatory_constraints,
        initial_policy: np.ndarray # should be in a cellular encoding
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
                self.reward_function[flat_state,flat_action] = reward_function(self._unflatten(flat_state=flat_state), self._unflatten(flat_action=flat_action))

        # initialise behaviour policy
        self.behaviour_policy = initial_policy # using some kind of conversion?
        self.target_policy = np.zeros((self.n_cells, self.n_states), dtype=int)

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
        self.side_effects_functions = [{'safe', 'unsafe', 'silent'} for _ in range(n_intracellular_states)]
        self.intracellular_transition_estimates = np.zeros((self.n_intracellular_states, self.n_intracellular_actions, self.n_intracellular_states))
        self.intracellular_transition_errors = np.zeros((self.n_intracellular_states, self.n_intracellular_actions))
        self.intracellular_transition_indicators = np.ones((self.n_intracellular_states, self.n_intracellular_actions))
        self.transition_estimates = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.transition_errors = np.zeros((self.n_states, self.n_actions))

        # miscellaneous initialisations
        self.previous_state = None
        self.current_state = None
        self.action_sampled = False


    def sample_action(
        self,
        previous_state,
    ):

        """Sample an action from the behaviour policy and assign current state to previous state"""

        # assert that we put in the right state
        self.action_sampled = True

        # update state, action
        self.previous_state = self.cellular_encoding(previous_state)
        self.flat_previous_state = self._flatten(state=self.previous_state)
        self.action = self.behaviour_policy[:, self.flat_previous_state]
        
        # return action
        return self.cellular_decoding(self.action)


    def update(
        self,
        current_state,
        reward,
        side_effects=None,
    ):

        """Update the agent's policy and statistics"""

        assert self.action_sampled # avoid double updating
        self.action_sampled = False
        self.current_state = self.cellular_encoding(current_state)
        self.flat_action = self._flatten(state=self.action)
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
        next_action = self.behaviour_policy[:, self._flatten(state=self.current_state)]
        new_episode = (self.current_episode_count[self._flatten(state=self.current_state), self._flatten(action=next_action)] >= max(1, self.previous_episodes_count[self._flatten(self.current_state), self._flatten(next_action)]))
        if new_episode:
            self._update_confidence_sets()
            self._extended_value_iteration()
            self._pe_shield()
            #self.current_episode_time_step = self.time_step # is this right? If not, could I just use the time step?
            self.previous_episodes_count += self.current_episode_count
            self.cellular_previous_episodes_count += self.cellular_current_episode_count

    # subroutines

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
            self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], self.current_state[cell]] = self.intracellular_transition_sum[self.previous_state[cell], self.action[cell], self.current_state[cell]] / self.intracellular_sum[self.previous_state[cell], self.action[cell]]
            # prune
            self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], :]  *= self.intracellular_transition_indicators[self.previous_state[cell], self.action[cell]]
        
        flat_previous_state = self._flatten(state=self.previous_state)
        flat_action = self._flatten(action=self.action)
        flat_current_state = self._flatten(state=self.current_state)
        self.transition_estimates[flat_previous_state, flat_action, flat_current_state] = np.prod(
            [self.intracellular_transition_estimates[self.previous_state[cell], self.action[cell], self.current_state[cell]] for cell in range(self.n_cells)]
        )

        # update errors
        self.transition_errors[flat_previous_state, flat_action] = np.sqrt(
            (
                14 * self.n_states * np.log(2 * self.n_actions * self.time_step)
            ) / (
                max([1, self.cellular_previous_episodes_count[flat_previous_state, flat_action]])
            )
        )

    def _flatten(
        self,
        state=None,
        action=None,
    ):

        assert (state is None) ^ (action is None)
        if type(state) is np.ndarray:
            n = self.n_intracellular_states
            obj = state
        elif type(action) is np.ndarray:
            n = self.n_intracellular_actions
            obj = action
        else:
            raise ValueError('state or action must be an array')

        bit_length = (n - 1).bit_length()
        bin_array = np.zeros((self.n_cells, bit_length))
        for cell in range(self.n_cells):
            bin_string = bin(obj[cell])[2:]
            start_bit = bit_length - len(bin_string)
            for bit in range(start_bit, bit_length):
                bin_array[cell, bit] = int(bin_string[bit - start_bit])
        bin_list = np.reshape(bin_array, np.prod(np.shape(bin_array)))#np.ravel(bin_array)
        integer = int(np.dot(np.flip(bin_list), 2 ** np.arange(bin_list.size)))
        return integer

    
    def _unflatten(
        self,
        flat_state=None,
        flat_action=None,
    ):

        assert (flat_state is None) ^ (flat_action is None)
        if type(flat_state) is int:
            n = self.n_intracellular_states
            obj = flat_state
        elif type(flat_action) is int:
            n = self.n_intracellular_actions
            obj = flat_action
        else:
            raise ValueError('flat_state or flat_action must be an integer')

        binary = bin(obj)[2:]
        bit_length = (n * self.n_cells - 1).bit_length()
        bin_list = np.zeros(bit_length)
        start_bit = bit_length - len(binary)
        for bit in range(start_bit, bit_length):
            bin_list[bit] = int(binary[bit - start_bit])
        bin_array = np.reshape(bin_list, (self.n_cells, int(bit_length / self.n_cells)))
        int_list = np.zeros(self.n_cells, dtype=int)
        for cell in range(self.n_cells):
            int_list[cell] = np.dot(np.flip(bin_array[cell]), 2 ** np.arange(bin_array[cell].size))
        return int_list


    def _inner_max(
        self,
        flat_state,
        flat_action,
        value,
    ):

        #flat_state = self._flatten(state=state)
        #flat_action = self._flatten(action=action)
        sorted_states = np.argsort(value)
        max_p = np.zeros(self.n_states)
        for flat_next_state in range(self.n_states):
            max_p[flat_next_state] = self.transition_estimates[flat_state,flat_action,flat_next_state]
        #x = self.transition_estimates[flat_state,flat_action,sorted_states[-1]] + self.transition_errors[flat_state, flat_action]
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
                max_flat_action = int(np.argmax(values))
                max_action = self._unflatten(flat_action=max_flat_action)
                self.target_policy[:, flat_state] = max_action
            max_diff = max([current_value[flat_state] - previous_value[flat_state] for flat_state in range(self.n_states)])
            min_diff = min([current_value[flat_state] - previous_value[flat_state] for flat_state in range(self.n_states)])
            stop = (max_diff - min_diff < self.accuracy)
            previous_value = current_value

    def _pe_shield(self):
        
        tmp_policy = self.behaviour_policy
        cell_set = set(range(self.n_cells))
        while len(cell_set) >= 1:
            cell = self._cell_prioritisation(cell_set)
            cell_set -= set(cell)
            tmp_policy[cell, :] = self.target_policy[cell, :]
            verified = self._verify(tmp_policy)
            if not verified:
                tmp_policy[cell, :] = self.behaviour_policy[cell, :]
        self.behaviour_policy = tmp_policy

    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        cell = random.sample(cell_set, 1)
        return cell

    def _verify(
        self,
        tmp_policy,
    ):

        # here I reshape the data and call an external verification framework

        return True




