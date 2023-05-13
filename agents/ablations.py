from .peucrl import PeUcrlAgt

import copy as cp

class NoShieldAgt(PeUcrlAgt):

    def name(self):
        return 'no shielding'

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        self.policy = cp.copy(target_policy)

class NoPruningAgt(PeUcrlAgt):

    def name(self):
        return 'no action-pruning'

    def action_pruning(self):
        self.new_pruning = False

class UnsafeBaselineAgt(PeUcrlAgt):

    def name(self):
        return 'unsafe baseline'

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):
        self.policy = cp.copy(target_policy)

    def action_pruning(self):
        self.new_pruning = False

    def reward_shaping(self, tabular_state, tabular_action):
        return 0.0