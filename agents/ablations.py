from .peucrl import PeUcrlAgent as PeUcrl
from copy import deepcopy


class NoShielding(PeUcrl):

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):

        self.policy = deepcopy(target_policy)


class NoShieldingNoPruning(NoShielding):

    def action_pruning(self):

        self.new_pruning = False