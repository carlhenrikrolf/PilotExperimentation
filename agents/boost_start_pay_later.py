from .peucrl import *
from .ablations import *

import numpy as np

def boosted_distances(agt):
    for s in range(agt.prior_knowledge.n_states):
        for a in range(agt.prior_knowledge.n_actions):
            maxN = max([1, agt.Nk[s, a]])
            agt.r_distances[s, a] = 1.0 / maxN
            if agt.prior_knowledge.identical_intracellular_transitions is True:
                maxN = max([1, agt.transferNk[s, a]])
            agt.p_distances[s, a] = 1.0 / maxN + np.sqrt(1.0/maxN)

            # dampened exploration:
            agt.r_distances[s, a] *= 1.0 / maxN
            agt.r_distances[s, a] = np.sqrt(agt.r_distances[s, a])

class BoostStartPeUcrlAgt(PeUcrlAgt):

    def name(self):
        return 'PE-UCRL with boosted start'

    def distances(self):
        boosted_distances(self)

class BoostStartNoShieldAgt(NoShieldAgt):

    def name(self):
        return 'No shielding with boosted start'

    def distances(self):
        boosted_distances(self)

class NonCellularBoostStartNoShieldAgt(BoostStartNoShieldAgt):

    def name(self):
        return 'Non-cellular PE-UCRL with boosted start'
    
    def reset_seed(self):
        super().reset_seed()
        self.prior_knowledge.identical_intracellular_transitions = False