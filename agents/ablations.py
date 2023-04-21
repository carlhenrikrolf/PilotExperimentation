from .peucrl import PeUcrlAgent
from copy import deepcopy


class NoShieldingAgent(PeUcrlAgent):

    def pe_shield(self, behaviour_policy, target_policy, p_estimate):

        self.policy = deepcopy(target_policy)


class NoShieldingNoPruningAgent(NoShieldingAgent):

    def action_pruning(self):

        self.start_time_step = 0
        self.end_time_step = 0

        self.new_pruning = False