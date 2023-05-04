"""Copypasted and modified, cite"""

from agents.utils import *
from gym_cellular.envs.utils import generalized_cellular2tabular as cellular2tabular, generalized_tabular2cellular as tabular2cellular

import copy as cp
import numpy as np
import os
import subprocess
from time import perf_counter

class PeUcrlAgt:

    def __init__(
        self,
        seed,
        prior_knowledge,
        regulatory_constraints,
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
        self.p_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.Pk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions, self.prior_knowledge.n_states),
            dtype=int,
        )
        self.u = np.zeros(
            shape=self.prior_knowledge.n_states,
            dtype=float,
        )
        if hasattr(prior_knowledge, 'reward_func'):
            self.reward_func = np.zeros(
                shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
                dtype=float,
            )
            for s in range(self.prior_knowledge.n_states):
                for a in range(self.prior_knowledge.n_actions):
                    self.reward_func[s, a] = self.prior_knowledge.reward_func(
                        state=self.prior_knowledge.detabularize(
                            s,
                            self.prior_knowledge.state_space,
                        ),
                        action=self.prior_knowledge.detabularize(
                            a,
                            self.prior_knowledge.action_space,
                        ),
                        next_state=(0, 0), # this needs to change
                    )
        self.r_distances = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )
        self.Rk = np.zeros(
            shape=(self.prior_knowledge.n_states, self.prior_knowledge.n_actions),
            dtype=float,
        )


        # Misc initializations
        initial_cellular_state = self.prior_knowledge.cellularize(
            element=self.prior_knowledge.initial_state,
            space=self.prior_knowledge.state_space,
        )
        self.last_cellular_state = initial_cellular_state
        self.initial_tabular_state = cellular2tabular(
            initial_cellular_state,
            self.prior_knowledge.state_space,
        )
        self.last_tabular_state = cp.copy(self.initial_tabular_state)
        self.current_tabular_state = cp.copy(self.initial_tabular_state)
        self.initial_policy = np.zeros(
            shape=(self.prior_knowledge.n_cells, self.prior_knowledge.n_states),
            dtype=int,
        ) # policy
        for s in range(self.prior_knowledge.n_states):
            S = self.prior_knowledge.detabularize(
                tabular_element=s,
                space=self.prior_knowledge.state_space
            )
            A = self.prior_knowledge.initial_policy(S)
            self.initial_policy[:, s] = self.prior_knowledge.cellularize(
                element=A,
                space=self.prior_knowledge.action_space,
            )
        self.policy = cp.copy(self.initial_policy)
        self.policy_update = np.zeros(
            shape=self.prior_knowledge.n_cells,
            dtype=int,
        )
        self.side_effects_funcs = [{'safe', 'unsafe'} for _ in range(self.prior_knowledge.n_intracellular_states)]

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
                    if hasattr(self.prior_knowledge, 'reward_func'):
                        optimistic_reward = self.reward_func[s, a]
                    else:
                        optimistic_reward = min((1, r_estimate[s, a] + self.r_distances[s, a]))
                    temp[a] = optimistic_reward + sum(
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
        behaviour_policy = cp.copy(self.policy)
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))
        target_policy = cp.copy(self.policy)
        self.pe_shield(behaviour_policy, target_policy, p_estimate)


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
        self.side_effects_processing(side_effects)
        self.updatev()
        self.updateP()
        self.updateR()
        self.t += 1

    # Registering new side effects
    def side_effects_processing(self, side_effects):

        for reporting_cell in range(self.prior_knowledge.n_cells):
            for reported_cell in range(self.prior_knowledge.n_cells):
                reported_current_intracellular_state = self.current_cellular_state[reported_cell]
                if side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_funcs[reported_current_intracellular_state] -= {'unsafe'}
                elif side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_funcs[reported_current_intracellular_state] -= {'safe'}

    # Applying shielding
    def pe_shield(self, behaviour_policy, target_policy, p_estimate, epsilon=0.8):
        
        tmp_policy = cp.copy(behaviour_policy)
        cell_set = set(range(self.prior_knowledge.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            # random_reset = np.random.rand() <= epsilon and self.current_tabular_state == self.initial_tabular_state and self.policy_update[cell] == 1
            # if random_reset: # increase exploration
            #     tmp_policy[cell, :] = cp.copy(self.initial_policy[cell, :])
            #     self.policy_update[cell] = 0
            # else:
            tmp_policy[cell, :] = cp.copy(target_policy[cell, :])
            if self.policy_update[cell] == 0:
                initial_policy_is_updated = True
                self.policy_update[cell] = 1
            else:
                initial_policy_is_updated = False
            verified = self.verify_with_prism(tmp_policy, p_estimate)
            if not verified:
                tmp_policy[cell, :] = cp.copy(behaviour_policy[cell, :])
                if initial_policy_is_updated:
                    self.policy_update[cell] = 0
        self.policy = cp.copy(tmp_policy)

    # Prism
    def verify_with_prism(
        self,
        tmp_policy,
        p_estimate,
    ):
        
        # initialise prism
        tmp_id = np.random.randint(0, 1000000)
        self.prism_path = 'agents/prism_files/tmp_' + str(tmp_id) + '/'
        try:
            os.mkdir(self.prism_path)
        except FileExistsError:
            print("Error: Cannot create folder. Clean 'agents/prism_files/' folder.")
        with open(self.prism_path + 'constraints.props', 'a') as props_file:
            props_file.write(self.regulatory_constraints)

        # write model file
        self.write_model_file(tmp_policy, p_estimate)

        # verify
        try:
            output = subprocess.check_output(['prism/prism/bin/prism', self.prism_path + 'model.prism', self.prism_path + 'constraints.props'])
        except subprocess.CalledProcessError as error:
            print(error.output.decode())
            raise ValueError('Prism returned an error, see above.')
        output = output.decode()
        occurances = 0
        for line in output.splitlines():
            if 'Result:' in line:
                occurances += 1
                if 'true' in line:
                    verified = True
                elif 'false' in line:
                    verified = False
                else:
                    raise ValueError('Verification returned non-Boolean result.')
        if occurances != 1:
            raise ValueError('Verification returned ' + str(occurances) + ' results. Expected 1 Boolean result.')
        self.prism_output = output # for debugging purposes

        # clean
        os.system('rm -r -f ' + self.prism_path)

        return verified
    
    def write_model_file(
            self,
            tmp_policy,
            p_estimate,
            epsilon: float = 0.000000000000001,
        ):
        
        os.system('rm -fr ' + self.prism_path + 'model.prism')
        with open(self.prism_path + 'model.prism', 'a') as prism_file:

            prism_file.write('dtmc\n\n')

            for s in range(self.prior_knowledge.n_states):
                for cell in range(self.prior_knowledge.n_cells):
                    C = 0
                    if self.policy_update[cell] == 1:
                        state = tabular2cellular(
                            s,
                            self.prior_knowledge.state_space,
                        )
                        for si in state:
                            if 'unsafe' in self.side_effects_funcs[si]:
                                C = 1
                                break
                    prism_file.write('const int C' + str(s) + '_' + str(cell) + ' = ' + str(C) + ';\n')
            prism_file.write('\n')

            prism_file.write('module M\n\n')

            prism_file.write('s : [0..' + str(self.prior_knowledge.n_states) + '] init ' + str(self.last_tabular_state) + ';\n')
            for cell in range(self.prior_knowledge.n_cells):
                prism_file.write('c_' + str(cell) + ' : [0..1] init C' + str(self.last_tabular_state) + '_' + str(cell) + ';\n')
            prism_file.write('\n')

            for s in range(self.prior_knowledge.n_states):
                prism_file.write('[] (s = ' + str(s) + ') -> ')
                a = cellular2tabular(
                    tmp_policy[:, s], 
                    self.prior_knowledge.action_space,
                )
                init_iter = True
                for next_s in range(self.prior_knowledge.n_states):
                    lb = max(
                        [epsilon,
                         p_estimate[s, a, next_s] - self.p_distances[s, a]]
                    )
                    ub = min(
                        [1-epsilon,
                         p_estimate[s, a, next_s] + self.p_distances[s, a]]
                    )
                    if not init_iter:
                        prism_file.write(' + ')
                    prism_file.write('[' + str(lb) + ',' + str(ub) + "] : (s' = " + str(next_s) + ')')
                    for cell in range(self.prior_knowledge.n_cells):
                        prism_file.write(' & (c_' + str(cell) + "' = C" + str(next_s) + '_' + str(cell) + ')')
                    init_iter = False
                prism_file.write(';\n')
            prism_file.write('\n')

            prism_file.write('endmodule\n\n')

            prism_file.write("formula n = ")
            for cell in range(self.prior_knowledge.n_cells):
                prism_file.write("c_" + str(cell)+ " + ")
            prism_file.write("0;\n")
            for count, cell_class in enumerate(self.prior_knowledge.cell_classes):
                prism_file.write("formula n_" + cell_class + " = ")
                for cell in self.prior_knowledge.cell_labelling_function[count]:
                    prism_file.write("c_" + str(cell) + " + ")
                prism_file.write("0;\n")

    # To get the data to save.
    def get_data(self):
        return self.data

