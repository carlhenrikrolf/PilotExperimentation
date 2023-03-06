from .peucrl import PeUcrlAgent
import numpy as np

class PeUcrlMinusRAgent(PeUcrlAgent):

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
        reward_function: np.ndarray, #### does not matter
    ):
        
        def reward_function(flat_state, flat_action):
            return 0
        super().__init__(
            confidence_level=confidence_level,
            accuracy=accuracy,
            n_cells=n_cells,
            n_intracellular_states=n_intracellular_states,
            cellular_encoding=cellular_encoding,
            n_intracellular_actions=n_intracellular_actions,
            cellular_decoding=cellular_decoding,
            cell_classes=cell_classes,
            cell_labelling_function=cell_labelling_function,
            regulatory_constraints=regulatory_constraints,
            initial_policy=initial_policy,
            reward_function=reward_function,
        )
        self.reward_sum = np.zeros((self.n_states, self.n_actions), dtype = float)
        self.reward_estimates = np.zeros((self.n_states, self.n_actions), dtype = float)
        self.reward_errors = np.zeros((self.n_states, self.n_actions), dtype = float) 


    def update(self, state, reward, side_effects):
        flat_action = self._flatten(action=self.action)
        self.reward_sum[self.flat_previous_state, flat_action] += reward
        super().update(state, reward, side_effects)

    def _extended_value_iteration(self):

        for flat_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                self.reward_estimates[flat_state,flat_action] = self.reward_sum[flat_state, flat_action] / max([1, self.previous_episodes_count[flat_state, flat_action]])
                self.reward_errors[flat_state, flat_action] = np.sqrt(
                    (
                        7 * np.log(2 * self.n_states * self.n_actions * self.time_step / self.confidence_level) 
                    ) / (
                        2 * max(
                            [1,
                            self.previous_episodes_count[flat_state, flat_action]]
                        )
                    )
                )
                self.reward_function[flat_state, flat_action] = min(
                    [1,
                    self.reward_estimates[flat_state, flat_action] + self.reward_errors[flat_state, flat_action]]
                )
        super()._extended_value_iteration()