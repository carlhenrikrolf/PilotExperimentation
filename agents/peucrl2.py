from .ucrl2 import Ucrl2Agent as Ucrl2
from .utils import cellular2tabular, tabular2cellular

from copy import deepcopy
import numpy as np
from os import system
from psutil import Process
import subprocess

class PeUcrl(Ucrl2):

    def __init__(
        self,
        confidence_level,
        n_cells,
        n_intracellular_states,
        cellular_encoding,
        n_intracellular_actions,
        cellular_decoding,
        cell_classes,
        cell_labelling_function,
        regulatory_constraints,
        initial_policy,
        reward_function,
        seed=0,
    ):
        
        # change name to be more inline?
        self.confidence_level = confidence_level
        self.n_cells = n_cells
        self.n_intra_states = n_intracellular_states
        self.n_intra_actions = n_intracellular_actions
        self.cell_classes = cell_classes
        self.cell_labelling_function = cell_labelling_function
        self.regulatory_constraints = regulatory_constraints
        # use tabular encoding for ucrl2
        self.cellular_encoding = cellular_encoding
        self.cellular_decoding = cellular_decoding

        # for cleaning later
        #tabular_encoding = lambda state: cellular2tabular(cellular_encoding(state), self.n_intra_states, self.n_cells)
        #tabular_decoding = lambda action: cellular2tabular(cellular_decoding(action), ...)
        
        super().__init__(
            self,
            confidence_level=self.confidence_level,
            n_cells=self.n_cells,
            n_intracellular_states=self.n_intra_states,
            cellular_encoding=self.cellular_encoding,
            n_intracellular_actions=self.n_intra_actions,
            cellular_decoding=self.cellular_decoding,
            initial_policy=initial_policy, # not self
            seed=seed, # not self
        )

        self.n_states = self.nS
        self.n_actions = self.nA

        # initialise arrays
        self.cell_Nk = np.zeros(shape=[self.n_states, self.n_actions])
        # self.intra_trans_fn = np.zeros(shape=[self.n_intra_states, self.n_intra_actions, self.n_intra_states])
        # self.intra_trans_err = np.zeros(shape=[self.n_intra_states, self.n_intra_actions, self.n_intra_states])
        # self.intra_trans_est = np.zeros(shape=[self.n_intra_states, self.n_intra_actions])
        # self.intra_Nk = ...
        # self.intra_vk = ...
        # self.cellular_Nk = ...
        # self.cellular_vk = ...
        self.policy_update = np.zeros(shape=[self.n_cells], dtype=bool)

        # misc initialisations
        self.cpu_id = Process().cpu_num()

        # initialise prism
        self.prism_path = 'agents/prism_files/cpu_' + str(self.cpu_id) + '/'
        system('rm -r -f ' + self.prism_path + '; mkdir ' + self.prism_path)
        with open(self.prism_path + 'constraints.props', 'a') as props_file:
            props_file.write(self.regulatory_constraints)

    def updateN(self):
        super().updateN()
        for tab_s in range(self.n_states):
            for tab_a in range(self.n_actions):
                self.cell_Nk[tab_s, tab_a] += self.cell_vk[tab_s, tab_a]

    # def update_cellular_vk(self):
    #     for tabular_state in range(self.n_states):
    #         for intra_state in tabular2cellular(tabular_state, self.n_intra_states, self.n_cells):
    #             if intra_state == self.observations[0][-2]:
    #                 for tabular_action in range(self.n_actions):
    #                     for intra_action in tabular2cellular(tabular_action, self.n_intra_actions, self.n_cells):
    #                         if intra_action == self.observations[1][-1]:
    #                             self.cellular_vk[tabular_state, tabular_action] += 1

    def updateP(self):
        for tabular_state in range(self.n_states):
            for intra_state in tabular2cellular(tabular_state, self.n_intra_states, self.n_cells):
                if intra_state == self.observations[0][-2]:
                    for tabular_action in range(self.n_actions):
                        for intra_action in tabular2cellular(tabular_action, self.n_intra_actions, self.n_cells):
                            if intra_action == self.observations[1][-1]:
                                for tabular_next_state in range(self.n_states):
                                    for intra_next_state in tabular2cellular(tabular_next_state, self.n_intra_states, self.n_cells):
                                        if intra_next_state == self.observations[0][-1]:
                                            self.Pk[tabular_state, tabular_action, tabular_next_state] += 1


    def distances(self):

        for tab_s in range(self.n_states):
            for tab_a in range(self.n_actions):

                self.r_distances[tab_s, tab_a] = np.sqrt((7 * np.log(2 * self.n_states * self.n_actions * self.t / self.delta)) # self.confidence_level
                / (2 * max([1, self.Nk[tab_s, tab_a]])))

                self.p_distances[tab_s, tab_a] = np.sqrt((14 * self.n_states * np.log(2 * self.n_actions * self.t / self.delta))
                / (max([1, self.cell_Nk[tab_s, tab_a]]))) # cell_Nk
    
    def new_episode(self):
        behaviour_policy = deepcopy(self.policy)
        super().new_episode()
        target_policy = deepcopy(self.policy)
        self.policy = self.pe_shield(behaviour_policy, target_policy)

    def distances(self):
        ...

    def update(self, current_state, reward, side_effects):

        super().update(current_state, reward, side_effects)

        # intracellular
        cellular_state = tabular2cellular(self.observations[0][...], self.n_intracellular_states, self.n_cells)
        cellular_action = tabular2cellular(self.observations[1][-1], self.n_intracellular_actions, self.n_cells)
        for (intra_state, intra_action) in zip(cellular_state, cellular_action):
            self.intra_vk[intra_state, intra_action] += 1

        # cellular
        for tabular_state in range(self.n_states):
            cellular_state = tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells)
            for tabular_action in range(self.n_actions):
                cellular_action = tabular2cellular(tabular_action, self.n_intracellular_actions, self.n_cells)
                self.cellular_vk[tabular_state, tabular_action] = np.amin(
                    [
                        self.intra_vk[intra_state, intra_action] \
                        for intra_state, intra_action \
                        in zip(cellular_state, cellular_action)
                    ]
                )
    ############ shouldn't this be zip???
    # ok, I've fixed this, but not sure any major error ...
    # doesn't seem to matter for the trasnition estimates, only for the when to update the policy.

    # needs work
    def update_estimates(self):

        # update estimates
        for intracellular_state in range(self.n_intracellular_states):
            for intracellular_action in range(self.n_intracellular_actions):
                self.intracellular_transition_estimates[intracellular_state, intracellular_action, :] \
                = self.intracellular_transition_sum[intracellular_state, intracellular_action, :] \
                / max([1, self.intracellular_sum[intracellular_state, intracellular_action]]) # change to count?

        for flat_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                for flat_next_state in range(self.n_states):
                    self.transition_estimates[flat_state, flat_action, flat_next_state] = 1
                    for (intracellular_state, intracellular_action, intracellular_next_state) in zip(
                                tabular2cellular(flat_state, self.n_intracellular_states, self.n_cells),
                                tabular2cellular(flat_action, self.n_intracellular_actions, self.n_cells),
                                tabular2cellular(flat_next_state, self.n_intracellular_states, self.n_cells)
                        ):
                        self.transition_estimates[flat_state, flat_action, flat_next_state] \
                        *= self.intracellular_transition_estimates[intracellular_state, intracellular_action, intracellular_next_state]

    def pe_shield(self): # I think this one is done, but it is much more complex now ...
        
        tmp_cellular_policy = np.zeros(shape=[self.n_states, self.n_intra_actions, self.n_cells])
        for tabular_state in range(self.n_states):
            for tabular_action in range(self.n_actions):
                cellular_action = tabular2cellular(tabular_action, self.n_intra_actions, self.n_cells)
                for intra_action in cellular_action:
                    tmp_cellular_policy[tabular_state, intra_action, :] = deepcopy(
                        self.behaviour_policy[tabular_state, tabular_action] ** (1/self.n_cells)
                    )
        reverted_policy = deepcopy(self.behaviour_policy)
        reverted_cellular_policy = deepcopy(tmp_cellular_policy)
        cell_set = set(range(self.n_cells))
        while len(cell_set) >= 1:
            cell = np.random.choice(list(cell_set))
            cell_set -= {cell}
            for tabular_action in range(self.n_actions):
                cellular_action = tabular2cellular(tabular_action, self.n_intra_actions, self.n_cells)
                for intra_action in cellular_action:
                    tmp_cellular_policy[:, intra_action, cell] = deepcopy(
                        self.target_policy[:, tabular_action] ** (1/self.n_cells)
                    )
            tmp_policy = np.zeros(shape=[self.n_states, self.n_actions])
            for tabular_state in range(self.n_states):
                for tabular_action in range(self.n_actions):
                    cellular_action = tabular2cellular(tabular_action, self.n_intra_actions, self.n_cells)
                    tmp_policy[tabular_state, tabular_action] = np.prod(
                        [
                            tmp_cellular_policy[tabular_state, intra_action, cell] \
                            for intra_action, cell \
                            in zip(cellular_action, range(self.n_cells))
                        ]
                    )
            if not self.policy_update[cell]:
                initial_policy_is_updated = True
                self.policy_update[cell] = True
            else:
                initial_policy_is_updated = False
            verified = self.verify_with_prism(tmp_policy)
            if not verified:
                tmp_policy = deepcopy(reverted_policy)
                tmp_cellular_policy = deepcopy(reverted_cellular_policy)
                if initial_policy_is_updated:
                    self.policy_update[cell] = False
            else:
                reverted_policy = deepcopy(tmp_policy)
                reverted_cellular_policy = deepcopy(tmp_cellular_policy)
        return tmp_policy




    # using prism

    def verify_with_prism(
        self,
        tmp_policy,
    ):
        
        self.write_prism_file(tmp_policy)
        try:
            output = subprocess.check_output(['prism/prism/bin/prism', self.prism_path + 'model.prism', self.prism_path + 'constraints.props'])
        except subprocess.CalledProcessError as error:
            error = error.output
            error = error.decode()
            print(error)
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

        return verified
    

    
    def write_prism_file(
            self,
            tmp_policy,
            epsilon: float = 0.000000000000001,
        ):
        
        system('rm -fr ' + self.prism_path + 'model.prism')
        with open(self.prism_path + 'model.prism', 'a') as prism_file:

            prism_file.write('dtmc\n\n')

            for tabular_state in range(self.n_states):
                for cell in range(self.n_cells):
                    C = 0
                    if self.policy_update[cell]:
                        cellular_state = tabular2cellular(tabular_state, self.n_intracellular_states, self.n_cells)
                        for intra_state in cellular_state:
                            if 'unsafe' in self.side_effects_functions[intra_state]:
                                C = 1
                                break
                    prism_file.write('const int C' + str(tabular_state) + '_' + str(cell) + ' = ' + str(C) + ';\n')
            prism_file.write('\n')

            prism_file.write('module M\n\n')

            prism_file.write('s : [0..' + str(self.n_states) + '] init ' + str(self.observations[0][-1]) + ';\n') ## Check what observation should be here
            for cell in range(self.n_cells):
                prism_file.write('c_' + str(cell) + ' : [0..1] init C' + str(self.observations[0][-1]) + '_' + str(cell) + ';\n') ## likewise
            prism_file.write('\n')

            for tabular_state in range(self.n_states):
                prism_file.write('[] (s = ' + str(tabular_state) + ') -> ')
                init_iter = True
                for next_tabular_state in range(self.n_states):
                    lb = max(
                        [
                            epsilon,
                            np.dot(
                                self.transition_estimates[tabular_state, :, next_tabular_state] \
                                - self.transition_errors[tabular_state,:],
                                tmp_policy[tabular_state, :]
                            )
                        ]
                    )
                    ub = min(
                        [
                            1-epsilon,
                            np.dot(
                                self.transition_estimates[tabular_state, :, next_tabular_state] \
                                + self.transition_errors[tabular_state, :],
                                tmp_policy[tabular_state]
                            )
                        ]
                    )
                    if not init_iter:
                        prism_file.write(' + ')
                    prism_file.write('[' + str(lb) + ',' + str(ub) + "] : (s' = " + str(next_tabular_state) + ')')
                    for cell in range(self.n_cells):
                        prism_file.write(' & (c_' + str(cell) + "' = C" + str(next_tabular_state) + '_' + str(cell) + ')')
                    init_iter = False
                prism_file.write(';\n')
            prism_file.write('\n')

            prism_file.write('endmodule\n\n')

            prism_file.write("formula n = ")
            for cell in range(self.n_cells):
                prism_file.write("c_" + str(cell)+ " + ")
            prism_file.write("0;\n")
            for count, cell_class in enumerate(self.cell_classes):
                prism_file.write("formula n_" + cell_class + " = ")
                for cell in self.cell_labelling_function[count]:
                    prism_file.write("c_" + str(cell) + " + ")
                prism_file.write("0;\n")


    ############ for later ############

    def action_pruning(self):
        ...
    
    def side_effects_processing(self):
        ...