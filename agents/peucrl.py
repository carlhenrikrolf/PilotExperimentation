import shared_subroutines
import numpy as np
import random
from itertools import chain
import linear programming software

class PeUcrlAgent:

    def __init__(
        self,
        # prior knowledge
        #intracellular_states, # intracellular_states maps states to an array of intracellular states
        #intracellular_actions, # similar for actions
        intracellular_mapping,
        initial_policy,
        confidence_level: float,
        reward_function=None,
    ):

        assert 0 < confidence_level < 1

        self.n_cells = len(intracellular_mapping(0))
        self.reward_function = reward_function
        self.behaviour_policy = initial_policy
        self.intracellular_mapping = intracellular_mapping

        # initialise counts to 0
        self.labelling_functions = [{'safe', 'unsafe', 'silent'} for _ in range(n_intracellular_states)]
        self.time_step = 0
        self.current_episode_time_step = 0
        self.current_episode_count = np.zeros((n_states, n_actions))
        self.previous_episodes_count = np.zeros((n_states, n_actions))
        self.cellular_current_episode_count = np.zeros((n_states, n_actions))
        self.cellular_previous_episodes_count = np.zeros((n_states, n_actions))
        self.transition_sum = np.zeros((n_states, n_actions, n_states))
        if self.reward_function != None:
            self.reward_sum = np.zeros(n_states, n_actions)

        # initialise statistics
        self.transition_estimates = np.zeros((n_states, n_actions, n_states))
        self.intracellular_transition_estimates = np.zeros((n_intracellular_states, n_intracellular_actions, n_intracellular_states))
        self.transition_errors = np.zeros((n_states, n_actions))
        self.intracellular_transition_errors = np.zeros((n_intracellular_states, n_intracellular_actions))
        self.transition_indicators = np.ones((n_states, n_actions))
        self.intracellular_transition_indicators = np.ones((n_intracellular_states, n_intracellular_actions))



    def sample_action(
        self,
        state,
    ):

        action = self.behaviour_policy(state)
        return action

    def update(
        self,
        state,
        action,
        next_state,
        reward,
        side_effects=None,
    ):

        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.intracellular_states = self.intracellular_mapping(state=state)
        self.intracellular_actions = self.intracellular_mapping(action=action)
        self.next_intracellular_states = self. intracellular_mapping(state=next_state)
        if side_effects == None:
            print("Warning: No side effects providing, assuming silence")
            side_effects = np.array([['silent' for _ in range(self.n_cells)] for _ in range(self.n_cells)])
        self.side_effects = side_effects

        # on-policy
        self.current_episode_count[state, action] += 1
        self._update_cellular_current_episode_count(state, action)
        self._side_effects_processing(next_state, side_effects)
        self._action_pruning(state, action, next_state)
        self.time_step += 1

        # off-policy
        new_episode = (self.current_episode_count >= max(1, self.previous_epsiodes_count))
        if new_episode:
            self._update_confidence_sets()
            self._extended_value_iteration() # available via pip3?
            self._pe_shield()
            self.episode += 1
            self.current_episode_time_step = self.time_step
            self.previous_episodes_count += self.current_episode_count
            self.cellular_previous_episodes_count += self.cellular_current_episode_count

    # subroutines (not shared)

    def _update_cellular_current_episode_count(self):

        for intracellular_state in self.intracellular_states:
            for intracellular_action in self.intracellular_actions:
                self.intracellular_episode_count[intracellular_state, intracellular_action] += 1
        self.cellular_current_episode_count[self.state, self.action] = np.amin(
            [
                [
                    self.intracellular_episode_count[intracellular_state, intracellular_action] for intracellular_state in self.intracellular_states
                ] for intracellular_action in self.intracellular_actions
            ]
        )

    def _side_effects_processing(self):

        for reporting_cell in range(self.n_cells):
            for reported_cell in range(self.n_cells):
                if self.side_effects[reporting_cell, reported_cell] == 'safe':
                    self.labelling_functions[self.intracellular_states[reported_cell]] -= {'unsafe'}
                elif self.side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.labelling_functions[self.intracellular_states[reported_cell]] -= {'safe'}



    def _action_pruning(self):

        # basic case
        for cell in range(self.n_cells):
            if 'unsafe' in self.labelling_functions[self.next_intracellular_states[cell]]:
                self.intracellular_transition_indicators[self.intracellular_states[cell], self.intracellular_actions[cell]] = 0
        
        # corner cases
        if self.time_step == 0:
            self.path = [set() for _ in range(self.n_cells)]
        for cell in range(self.n_cells):
            pruned_action_space = sum([self.intracellular_transition_indicators[self.intracellular_states[cell], intracellular_actions] for intracellular_actions in range(self.n_intracellular_actions)])
            if pruned_action_space >= 2:
                self.path[cell] = set()
            elif pruned_action_space == 1:
                self.path[cell].add([self.intracellular_states[cell], self.intracellular_actions[cell]])
            if ('unsafe' in self.labelling_functions[self.next_intracellular_states[cell]]) or pruned_action_space == 0:
                for [intracellular_state, intracellular_action] in self.path[cell]:
                    self.intracellular_transition_indicator[intracellular_state, intracellular_action] = 0

        # intra to extra # SEEMS EXTREMELY INEFFICIENT
        for state in range(self.n_states):
            for action in range


    def _update_confidence_sets(self):

        # update transition count
        self.transition_count # += 1 bl bla bla

        # update estimates
        self.transition_estimates[self.state, self.action, self.next_state] = (
            self.transition_count / (
                self.cellular_previous_episodes_count[self.state, self.action] + self.cellular_current_episode_count[self.state, self.action]
            )
        )

        # update errors
        self.transition_errors[self.state, self.action] = np.sqrt(
            (
                14 * self.n_states np.log(2 * self.n_actions * self.current_episode_time_step)
            ) / (
                max({1, self.previous_cellular_episode_count[self.state, self.action]})
            )
        )

        if self.reward_function == None:
            () #############
    
    def _max_probability(self, state, action, sorted_states):
        indicator = 1 # this indicator is our modification
        for intracellular_state in self.intracellular_mapping(state):
            for intracellular_action in self.intracellular_mapping(action):
                if self.transition_indicators[intracellular_state, intracellular_action] == 0:
                    indicator = 0
                    break # can we reorder to make this faster?
            if indicator == 0:
                break
        min1 = min([1, indicator * (self.transition_estimates[state,action] + self.transition_errors[state, action])])
        max_p = np.zeros(self.n_states)
        for i in range(self.n_states):
            max_p[i] = indicator * self.transition_estimates[state,action,i]
        max_p[sorted_states[-1]] = min1
        l = -2
        while sum(max_p) > 1:
            max_p[sorted_states[l]] = max([0, 1 - sum([max_p[k] for k in chain(range(0, sorted_states[l]), range(sorted_states[l]+1, self.n_states))])])
            l -= 1
            if l < - self.n_states:
                print("Error: Probabilities do not sum up to 1")
                break
        return max_p
        
    def _extended_value_iteration(self):

        previous_value = np.zeros(self.n_states)
        current_value = previous_value
        condition = True
        n_iterations = 0
        while condition:
            n_iterations += 1
            sorted_states = np.argsort(current_value) # from min to max
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    p_max = self.max_probability(state, action, sorted_states)
                    tmp[action] = self.reward_function[state, action] + sum([ v * p for (v, p) in zip(previous_value, p_max)])
                current_value[state] = max([tmp[a] for a in range(self.n_actions)])
            

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
                    self.labelling_functions[self.intracellular_states[reported_cell]] -= {'unsafe'}
                elif self.side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.labelling_functions[self.intracellular_states[reported_cell]] -= {'safe'}
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

