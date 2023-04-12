"""Copypasted and modified, cite"""

import copy as cp
from agents.utils import * #from learners.discreteMDPs.utils import *

#from learners.discreteMDPs.AgentInterface import Agent

class Ucrl2Agent:

    def __init__(
        self,
        confidence_level: float, # a parameter
        n_cells: int,
        n_intracellular_states: int,
        cellular_encoding, # states to N^n
        n_intracellular_actions: int,
        cellular_decoding, # N^n to actions # the (size, coding) pair is equivalent to the state/action space
        initial_policy: np.ndarray,
        reward_function = None,
        cell_classes = None,
        cell_labelling_function = None,
        regulatory_constraints = None, 
        seed=0,
    ):
        """
        Vanilla UCRL2 based on "Jaksch, Thomas, Ronald Ortner, and Peter Auer. "Near-optimal regret bounds for reinforcement learning." Journal of Machine Learning Research 11.Apr (2010): 1563-1600."
        :param nS: the number of states
        :param nA: the number of actions
        :param delta:  confidence level in (0,1)
        """
        self.nS = n_intracellular_states ** n_cells
        self.nA = n_intracellular_actions ** n_cells
        self.t = 1
        self.delta = confidence_level

        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.n_intracellular_actions = n_intracellular_actions
        self.cellular_encoding = cellular_encoding
        self.cellular_decoding = cellular_decoding
        np.random.seed(seed=seed)

        self.observations = [[], [], []] # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros((self.nS, self.nA)) #the state-action count for the current episode k
        self.Nk = np.zeros((self.nS, self.nA)) #the state-action count prior to episode k

        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))

        self.u = np.zeros(self.nS)
        self.span = []
        self.policy = np.zeros((self.nS, self.nA)) # policy
        for s in range(self.nS):
            for a in range(self.nA):
                if cellular2tabular(initial_policy[:, s], self.n_intracellular_actions, self.n_cells) == a:
                    self.policy[s, a] = 1.0

    def name(self):
        return "UCRL2"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.nS * np.log(2 * self.nA * self.t / self.delta))
                                                 / (max([1, self.Nk[s, a]])))

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.nS)
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
        u1 = np.zeros(self.nS)
        sorted_indices = np.arange(self.nS)
        niter = 0
        while True:
            niter += 1
            for s in range(self.nS):

                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break


    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.updateN()
        self.vk = np.zeros((self.nS, self.nA))
        r_estimate = np.zeros((self.nS, self.nA))
        p_estimate = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.nS):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.span = [0]
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA
        self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def sample_action(self, previous_state):
        state = self.cellular_encoding(previous_state)
        state = cellular2tabular(state, self.n_intracellular_states, self.n_cells)
        if self.t == 1:
            self.reset(state)
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stoppping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        self.previous_state = state
        self.previous_action = action
        tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
        action = self.cellular_decoding(action)
        return tabular2cellular(action, self.n_intracellular_actions, self.n_cells)

    # To update the learner after one step of the current policy.
    def update(self, current_state, reward, side_effects):
        self.vk[self.previous_state, self.previous_action] += 1
        observation = self.cellular_encoding(current_state)
        observation = cellular2tabular(observation, self.n_intracellular_states, self.n_cells)
        self.observations[0].append(observation)
        self.observations[1].append(self.previous_action)
        self.observations[2].append(reward)
        self.updateP()
        self.updateR()
        self.t += 1


    # subroutines the user can call to collect data

    def get_ns_between_time_steps(self):
        return 3.14 #self.end_time_step - self.start_time_step
    
    def get_ns_between_episodes(self):
        return 3.14 #self.end_episode - self.start_episode