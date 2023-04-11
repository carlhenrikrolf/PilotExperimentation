from .peucrl import PeUcrlAgent
from .peucrl_ablations import PeUcrlMinusRAgent
from copy import deepcopy
import numpy as np

class PeUcrlPlusCellularStatsAgent(PeUcrlAgent):

    pass



class PeUcrlPlusCellularStatsMinusRAgent(PeUcrlMinusRAgent):

    def __init__(
        self,
        confidence_level: float, # a parameter
        n_cells: int, # prior knowledge from here and down
        n_intracellular_states: int,
        cellular_encoding, # states to N^n
        n_intracellular_actions: int,
        cellular_decoding, # N^n to actions # the (size, coding) pair is equivalent to the state/action space
        cell_classes: set,
        cell_labelling_function,
        regulatory_constraints,
        initial_policy: np.ndarray, # should be in a cellular encoding
        seed=0,
    ):
    
        super().__init__(
            confidence_level=confidence_level,
            n_cells=n_cells,
            n_intracellular_states=n_intracellular_states,
            cellular_encoding=cellular_encoding,
            n_intracellular_actions=n_intracellular_actions,
            cellular_decoding=cellular_decoding,
            cell_classes=cell_classes,
            cell_labelling_function=cell_labelling_function,
            regulatory_constraints=regulatory_constraints,
            initial_policy=initial_policy,
            seed=seed,
        )


        self.intracellular_transition_errors = np.zeros((self.n_intracellular_states, self.n_intracellular_actions))


    def _update_errors(self):

        super()._update_errors()

        # update errors for intracellular state--action pairs
        for intracellular_state in range(self.n_intracellular_states):
            for intracellular_action in range(self.n_intracellular_actions):
                self.intracellular_transition_errors[intracellular_state, intracellular_action] = np.sqrt(
                    (
                        14 * self.n_intracellular_states * np.log(2 * self.n_intracellular_actions * self.time_step / self.confidence_level)
                    ) / (
                        max([1, self.intracellular_episode_count[intracellular_state, intracellular_action]]) # double-check this?
                    )
                )

    # to be continued