import shared_subroutines
import numpy as np
import random
import linear programming software

class PeUcrlAgent:

    def __init__(
        self,
        # prior knowledge
        intracellular_states, # intracellular_states maps states to an array of intracellular states
        intracellular_actions, # similar for actions
        initial_policy,
        confidence_level: float,
        reward_function=None,
    ):

        assert 0 < confidence_level < 1
        assert len(intracellular_states(any_state)) == len(intracellular_actions(any_action))

        self.n_cells = len(intracellular_states(any_state))
        self.reward_function = reward_function
        self.behaviour_policy = initial_policy

        # initialise counts to 0
        self.time_step = 0
        self.current_episode_time_step = 0
        self.current_episode_count = np.zeros(n_states, n_actions)
        self.previous_epsiodes_count = np.zeros(n_states, n_actions)
        self.transition_sum = np.zeros(n_states, n_actions, n_states)
        if self.reward_function != None:
            self.reward_sum = np.zeros(n_states, n_actions)


    def sample_action(
        self,
        state,
    ):

        return action

    def update(
        self,
        state,
        action,
        next_state,
        reward,
        side_effects,
    ):

        self._side_effects_processing(side_effects)

        self.time_step += 1

        new_episode = (self.current_episode_count >= max(1, self.previous_epsiodes_count))

        if new_episode:
            
            update confidence set
            target_policy = extended value iteration # available via pip3?
            self.behaviour_policy = self._pe_shield(self.behaviour_policy, target_policy)
            self.episode += 1
            self.current_episode_time_step = self.time_step

    # subroutines (not shared)

    def _side_effects_processing(
        self,
        side_effects
    ):

    def _action_pruning(
        self,
        state,
        action,
        next_state,
    ):

        return pruned_confidence_set

    def _pe_shield(
        self,
        behaviour_policy,
        target_policy,
        for cell in cell_set:
            verified = verify(model, policy)
            if verified:
                return policy
    ):

    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        if self.n_cells == len(cell_set):
            self.cell_set = set(range(self.n_cells))

        cell = random.sample(self.cell_set, 1)
        self.cell_set -= set(cell)

        return cell

class AntiSilencingPeUcrlAgent(PeUcrlAgent):

    pass

    silence = np.array(self.n_cells)

    def _side_effects_processing(
        self,
        side_effects
    ):



    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        if self.n_cells == len(cell_set):
            self.cell_set = set(range(self.n_cells))

            sample
            return cell

        else: 

            something

            return cell

