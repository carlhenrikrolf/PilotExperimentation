from .peucrl import PeUcrlAgent
from copy import deepcopy

class PeUcrlMinusShieldAgent(PeUcrlAgent):

    def _pe_shield(self):
        self.behaviour_policy = deepcopy(self.target_policy)