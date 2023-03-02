from .peucrl_minus_r import PeUcrlMinusRAgent
from copy import deepcopy

class PeUcrlMinusRMinusShieldAgent(PeUcrlMinusRAgent):

    def _pe_shield(self):
        self.behaviour_policy = deepcopy(self.target_policy)