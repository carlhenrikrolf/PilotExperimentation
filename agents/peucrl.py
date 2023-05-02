"""Copypasted and modified, cite"""

from agents.utils import *
from gym_cellular.envs.utils import generalized_cellular2tabular as cellular2tabular, generalized_tabular2cellular as tabular2cellular

import copy as cp
import numpy as np
from time import perf_counter

class PeUcrlAgt:

    def __init__(
        self,
        seed,
        prior_knowledge,
        regulatory_constraints='true',
    ):
        """
        Implementation of PeUcrl.
        """

        # Storing the parameters
        np.random.seed(seed=seed)
        self.prior_knowledge = prior_knowledge
        self.regulatory_constraints = regulatory_constraints

        # Initialize counters
        self.t = 1
        self.vk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        ) #the state-action count for the current episode k
        self.Nk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        ) #the state-action count prior to episode k
        self.r_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.p_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.Pk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions, self.prior_knowledge.n_states),
            dtype=int,
        )
        self.Rk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.u = np.zeros(
            shape=self.prior_knowledge.n_states,
            dtype=float,
        )

        # Misc initializations
        initial_cellular_state = self.prior_knowledge.cellularize(
            element=self.prior_knowledge.initial_state,
            space=self.prior_knowledge.state_space,
        )
        self.last_cellular_state = initial_cellular_state
        initial_tabular_state = cellular2tabular(
            initial_cellular_state,
            self.prior_knowledge.state_space,
        )
        self.last_tabular_state = initial_tabular_state
        self.current_tabular_state = initial_tabular_state
        self.policy = np.zeros(
            shape=(self.prior_knowledge.n_cells, self.prior_knowledge.n_states),
            dtype=int,
        ) # policy
        for s in range(self.prior_knowledge.n_states):
            S = self.prior_knowledge.detabularize(
                tabular_element=s,
                space=self.prior_knowledge.state_space
            )
            A = self.prior_knowledge.initial_policy(S)
            self.policy[:, s] = self.prior_knowledge.cellularize(
                element=A,
                space=self.prior_knowledge.action_space,
            )

        self.data = {}


    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                self.Nk[s, a] += self.vk[s, a]

    # Auxiliary function to update v the accumulated state-action count.
    def updatev(self):
        self.vk[self.last_tabular_state, self.last_tabular_action] += 1

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.last_tabular_state, self.last_tabular_action] += self.current_reward

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.last_tabular_state, self.last_tabular_action, self.current_tabular_state] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.prior_knowledge.n_states * self.prior_knowledge.n_actions * self.t / self.prior_knowledge.confidence_level))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.prior_knowledge.n_states * np.log(2 * self.prior_knowledge.n_actions * self.t / self.prior_knowledge.confidence_level))
                                                 / (max([1, self.Nk[s, a]])))

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.prior_knowledge.n_states)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1: 
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  # Error? 
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.prior_knowledge.n_states)
        sorted_indices = np.arange(self.prior_knowledge.n_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.prior_knowledge.n_states):

                temp = np.zeros(self.prior_knowledge.n_actions)
                for a in range(self.prior_knowledge.n_actions):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                sampled_cellular_action = tabular2cellular(
                    np.random.choice(choice),
                    self.prior_knowledge.action_space,
                )
                self.policy[:, s] = cp.copy(sampled_cellular_action)

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.prior_knowledge.n_states)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break


    # To start a new episode (init var, computes estmates and run EVI).
    def off_policy(self):
        self.updateN()
        self.vk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=int,
        )
        r_estimate = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        p_estimate = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions, self.prior_knowledge.n_states),
            dtype=float,
        )
        for s in range(self.prior_knowledge.n_states):
            for a in range(self.prior_knowledge.n_actions):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.prior_knowledge.n_states):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))


    def sample_action(self, state):
        self.last_cellular_state = self.prior_knowledge.cellularize(
            element=state,
            space=self.prior_knowledge.state_space,
        )
        self.last_tabular_state = cellular2tabular(
            self.last_cellular_state,
            self.prior_knowledge.state_space,
        )
        assert self.last_tabular_state == self.current_tabular_state
        self.last_cellular_action = cp.copy(self.policy[:, self.last_tabular_state])
        self.last_tabular_action = cellular2tabular(
            self.last_cellular_action,
            self.prior_knowledge.action_space,
        )
        self.new_episode = self.vk[self.last_tabular_state, self.last_tabular_action] >= max([1, self.Nk[self.last_tabular_state, self.last_tabular_action]])
        self.data['off_policy_time'] = np.nan
        if self.new_episode:
            self.data['off_policy_time'] = perf_counter()
            self.off_policy()
            self.last_cellular_action = cp.copy(self.policy[:, self.last_tabular_state])
            self.last_tabular_action = cellular2tabular(
                self.last_cellular_action,
                self.prior_knowledge.action_space,
            )
        self.data['off_policy_time'] = perf_counter() - self.data['off_policy_time']
        output = self.prior_knowledge.decellularize(
            cellular_element=self.last_cellular_action,
            space=self.prior_knowledge.action_space,
        )
        return output

    # To update the learner after one step of the current policy.
    def update(self, state, reward, side_effects):
        self.current_cellular_state = self.prior_knowledge.cellularize(
            element=state,
            space=self.prior_knowledge.state_space,
        )
        self.current_tabular_state = cellular2tabular(
            self.current_cellular_state,
            self.prior_knowledge.state_space,
        )
        self.current_reward = reward
        self.updatev()
        self.updateP()
        self.updateR()
        self.t += 1

    # To get the data to save.
    def get_data(self):
        return self.data

