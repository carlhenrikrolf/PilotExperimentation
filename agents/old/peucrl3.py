"""Copypasted and modified, cite"""

import copy as cp
from agents.utils import * #from learners.discreteMDPs.utils import *
from time import perf_counter_ns

#from learners.discreteMDPs.AgentInterface import Agent

# modifications:

from copy import deepcopy
import numpy as np
from os import system
from psutil import Process
import subprocess


class PeUcrlAgent:

    """This script adds transfer learning to UCRL2"""

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

        self.n_states = n_intracellular_states ** n_cells
        self.n_actions = n_intracellular_actions ** n_cells
        self.t = 1
        self.delta = confidence_level

        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.n_intracellular_actions = n_intracellular_actions
        self.cellular_encoding = cellular_encoding
        self.cellular_decoding = cellular_decoding
        np.random.seed(seed=seed)

        self.observations = [[], [], []] # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros((self.n_states, self.n_actions)) #the state-action count for the current episode k
        self.Nk = np.zeros((self.n_states, self.n_actions)) #the state-action count prior to episode k

        self.r_distances = np.zeros((self.n_states, self.n_actions))
        self.p_distances = np.zeros((self.n_states, self.n_actions))
        self.Pk = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.Rk = np.zeros((self.n_states, self.n_actions))

        self.u = np.zeros(self.n_states)
        self.span = []
        self.initial_policy = initial_policy
        self.policy = self.initial_policy
        # self.policy = np.zeros((self.n_states, self.n_actions)) # policy
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         if cellular2tabular(initial_policy[:, s], self.n_intracellular_actions, self.n_cells) == a:
        #             self.policy[s, a] = 1.0

        # modifications:

        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints
        self.cell_Nk = np.zeros(shape=[self.n_states, self.n_actions])
        self.policy_update = np.zeros(shape=[self.n_cells], dtype=bool)

    def name(self):
        return "PEUCRL"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Nk[s, a] += self.vk[s, a]
                self.cell_Nk[s, a] += self.cell_vk[s, a] # mod

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        for tabular_state in range(self.n_states):
            for intra_state in tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells):
                if intra_state == self.observations[0][-2]:
                    for tabular_action in range(self.n_actions):
                        for intra_action in tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells):
                            if intra_action == self.observations[1][-1]:
                                for tabular_next_state in range(self.n_states):
                                    for intra_next_state in tabular2cellular(tabular_next_state, self.n_intracellular_states, self.n_cells):
                                        if intra_next_state == self.observations[0][-1]:
                                            self.Pk[tabular_state, tabular_action, tabular_next_state] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.n_states * self.n_actions * self.t / self.delta))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.n_states * np.log(2 * self.n_actions * self.t / self.delta))
                                                 / (max([1, self.cell_Nk[s, a]]))) #mod

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.n_states)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1: 
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.n_states)
        sorted_indices = np.arange(self.n_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.n_states):

                temp = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg] # not modified for the sake of rewards
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                action = np.random.choice(choice)
                self.policy[:, s] = tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
                #self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.n_actions)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.n_states)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break


    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.start_episode = perf_counter_ns()
        self.updateN()
        self.vk = np.zeros((self.n_states, self.n_actions))
        self.cell_vk = np.zeros(shape=[self.n_states, self.n_actions])
        r_estimate = np.zeros((self.n_states, self.n_actions))
        p_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                div = max([1, self.cell_Nk[s, a]]) # mod
                r_estimate[s, a] = self.Rk[s, a] / max([1, self.Nk[s, a]]) # mod
                for next_s in range(self.n_states):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))
        self.end_episode = perf_counter_ns()

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.n_states, self.n_actions))
        self.Nk = np.zeros((self.n_states, self.n_actions))
        self.cell_vk = np.zeros((self.n_states, self.n_actions)) # mod
        self.cell_Nk = np.zeros((self.n_states, self.n_actions)) # mod
        self.u = np.zeros(self.n_states)
        self.Pk = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.Rk = np.zeros((self.n_states, self.n_actions))
        self.span = [0]
        self.policy = self.initial_policy
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         self.policy[s, a] = 1. / self.n_actions
        self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def sample_action(self, previous_state):
        s = self.cellular_encoding(previous_state)
        state = cellular2tabular(s, self.n_intracellular_states, self.n_cells)
        if self.t == 1:
            self.reset(state)
        action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
        #action = categorical_sample([self.policy[state, a] for a in range(self.n_actions)], np.random)
        self.start_episode = np.nan
        self.end_episode = np.nan
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):
            self.new_episode()
            action = cellular2tabular(self.policy[:,state], self.n_intracellular_actions, self.n_cells)
            #action = categorical_sample([self.policy[state, a] for a in range(self.n_actions)], np.random)
        self.previous_state = state
        self.previous_action = action
        tabular2cellular(action, self.n_intracellular_actions, self.n_cells)
        action = self.cellular_decoding(action)
        return tabular2cellular(action, self.n_intracellular_actions, self.n_cells)

    # modification:

    def updatev(self):
        self.vk[self.previous_state, self.previous_action] += 1 #standard
        for tabular_state in range(self.n_states):
            for intra_state in tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells):
                if intra_state == self.observations[0][-2]:
                    for tabular_action in range(self.n_actions):
                        for intra_action in tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells):
                            if intra_action == self.observations[1][-1]:
                                self.cell_vk[tabular_state, tabular_action] += 1


    # To update the learner after one step of the current policy.
    def update(self, current_state, reward, side_effects):
        observation = self.cellular_encoding(current_state)
        observation = cellular2tabular(observation, self.n_intracellular_states, self.n_cells)
        self.observations[0].append(observation)
        self.observations[1].append(self.previous_action)
        self.observations[2].append(reward)
        self.updatev()
        self.updateP()
        self.updateR()
        self.t += 1


    # subroutines the user can call to collect data

    def get_ns_between_time_steps(self):
        return 3.14 #self.end_time_step - self.start_time_step
    
    def get_ns_between_episodes(self):
        return self.end_episode - self.start_episode