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
        initial_policy, # should be in a cellular encoding
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
        self.reward_function = lambda flat_state, flat_action: reward_function(self._unflatten(flat_state=flat_state), self._unflatten(flat_action=flat_action))
        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints

        # compute additional parameters
        self.n_states = n_intracellular_states ** n_cells
        self.n_actions = n_intracellular_actions ** n_cells

        # initialise behaviour policy
        self.behaviour_policy = initial_policy # using some kind of conversion?
        self.target_policy = np.zeros(self.n_states)

        # initialise counts to 0
        self.time_step = 0
        self.current_episode_time_step = 0 # do I need this?
        self.current_episode_count = np.zeros((self.n_states, self.n_actions))
        self.previous_episodes_count = np.zeros((self.n_states, self.n_actions))
        self.cellular_current_episode_count = np.zeros((self.n_states, self.n_actions))
        self.cellular_previous_episodes_count = np.zeros((self.n_states, self.n_actions))
        self.intracellular_episode_count = np.zeros((self.n_intracellular_states, self.n_intracellular_actions))
        self.intracellular_sum = np.zeros((self.n_intracellular_states, self.n_intracellular_actions))
        self.intracellular_transition_sum = np.zeros((self.n_intracellular_states, self.n_intracellular_actions, self.n_intracellular_states))

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

        assert previous_state == self.previous_state or previous_state == self.current_state
        self.action_sampled = True

        # update state, action
        self.previous_state = self.cellular_encoding(previous_state)
        flat_state = self._flatten(self.previous_state)
        flat_action = self.behaviour_policy[flat_state]
        self.action = self._unflatten(flat_action=flat_action)
        
        # return action
        return self.cellular_decoding(self.action)


    def update(
        self,
        current_state,
        reward,
        side_effects=None,
    ):

        assert self.action_sampled # avoid double updating
        self.action_sampled = False
        self.current_state = self.cellular_encoding(current_state)
        self.reward = reward
        if side_effects == None:
            print("Warning: No side effects providing, assuming silence")
            side_effects = np.array([['silent' for _ in range(self.n_cells)] for _ in range(self.n_cells)])
        self.side_effects = side_effects

        # on-policy
        self.current_episode_count[self.previous_state, self.action] += 1
        self._side_effects_processing()
        self._action_pruning()
        self._update_current_episode_counts() # moved below. Correct?
        self.time_step += 1

        # off-policy
        new_episode = (self.current_episode_count >= max(1, self.previous_episodes_count))
        if new_episode:
            self._update_confidence_sets()
            self._extended_value_iteration()
            self._pe_shield()
            self.current_episode_time_step = self.time_step # is this right? If not, could I just use the time step?
            self.previous_episodes_count += self.current_episode_count
            self.cellular_previous_episodes_count += self.cellular_current_episode_count

    # subroutines

    def _update_current_episode_counts(self):

        # intracellular
        for intracellular_state in self.previous_state:
            for intracellular_action in self.action:
                self.intracellular_episode_count[intracellular_state, intracellular_action] += 1

        # cellular
        flat_previous_state = self._flatten(state=self.previous_state)
        flat_action = self._flatten(action=self.action)
        self.cellular_current_episode_count[flat_previous_state, flat_action] = np.amin(
            [
                [
                    self.intracellular_episode_count[intracellular_state, intracellular_action] for intracellular_state in self.previous_state
                ] for intracellular_action in self.action
            ]
        )

        # standard
        self.current_episode_count[flat_previous_state, flat_action] += 1
        

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
                14 * self.n_states * np.log(2 * self.n_actions * self.current_episode_time_step)
            ) / (
                max({1, self.cellular_previous_episodes_count[flat_previous_state, flat_action]})
            )
        )

    def _flatten(
        self,
        state=None,
        action=None,
    ):

        assert state == None ^ action == None
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
        bin_list = np.ravel(bin_array)
        integer = np.dot(np.flip(bin_list), 2 ** np.arange(bin_list.size))
        return integer

    
    def _unflatten(
        self,
        flat_state=None,
        flat_action=None,
    ):

        pass


    def _inner_max(
        self,
        state,
        action,
        value,
    ):

        flat_state = self._flatten(state=state)
        flat_action = self._flatten(action=action)
        sorted_states = np.argsort(value)
        max_p = np.zeros(self.n_states)
        for flat_next_state in range(self.n_states):
            max_p[flat_next_state] = self.transition_estimates[flat_state,flat_action,flat_next_state]
        max_p[sorted_states[-1]] = min(
            [
                1,
                self.transition_estimates[flat_state,flat_action] + self.transition_errors[flat_state, flat_action]
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
            if l < -self.n_states:
                raise ValueError('sum of max_p is greater than 1')
        return sum([ v * p for (v, p) in zip(value, max_p)])   


    def _extended_value_iteration(self):

        previous_value = np.zeros(self.n_states)
        current_value = np.zeros(self.n_states)
        stop = False
        while not stop:
            for flat_state in range(self.n_states):
                current_value[flat_state] = max([self.reward_function[flat_state, flat_action] + self._inner_max(flat_state, flat_action, previous_value) for flat_action in range(self.n_actions)])
                self.target_policy[flat_state] = np.argmax([self.reward_function[flat_state, flat_action] + self._inner_max(flat_state, flat_action, previous_value) for flat_action in range(self.n_actions)])
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
            tmp_policy[cell] = self.target_policy[cell]
            verified = self._verify(tmp_policy)
            if not verified:
                tmp_policy[cell] = self.behaviour_policy[cell]
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

class AntiSilencingPeUcrlAgent(PeUcrlAgent):

    pass

    silence = np.array(self.n_cells)

    def _side_effects_processing(
        self,
        next_state,
        side_effects,
    ):

        if self.time_step == 0: # change
            self.silence = np.zeros(self.n_cells)
        for reporting_cell in range(self.n_cells): # the same
            for reported_cell in range(self.n_cells):
                if self.side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_functions[self.intracellular_states[reported_cell]] -= {'unsafe'}
                elif self.side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_functions[self.intracellular_states[reported_cell]] -= {'safe'}
                else: # change
                    self.silence[reporting_cell] += 1


    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        if len(cell_set) == self.n_cells:
            self.last_cell = np.where(self.silence == self.silence.min())
            if len(self.last_cell) >= 2:
                self.last_cell = random.sample(self.last_cell, 1)
        if len(cell_set) >= 2:
            cell = random.sample(cell_set - self.last_cell)
        else:
            cell = self.last_cell
        return cell

