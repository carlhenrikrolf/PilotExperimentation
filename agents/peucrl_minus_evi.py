from .peucrl import PeUcrlAgent
from copy import deepcopy
import numpy as np

class PeUcrlMinusEviAgent(PeUcrlAgent):

    def _extended_value_iteration(self):
          
        self.target_policy = deepcopy(np.random.randint(0, self.n_intracellular_actions, size=(self.n_cells, self.n_states)))
