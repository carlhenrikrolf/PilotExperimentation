from .peucrl import PeUcrlAgent
from copy import deepcopy
import numpy as np




class PeUcrlMinusRAgent(PeUcrlAgent):

    def __init__(
        self,
        confidence_level: float, # a parameter
        accuracy: float, # a parameter
        n_cells: int, # prior knowledge from here and down
        n_intracellular_states: int,
        cellular_encoding, # states to N^n
        n_intracellular_actions: int,
        cellular_decoding, # N^n to actions # the (size, coding) pair is equivalent to the state/action space
        cell_classes: set,
        cell_labelling_function,
        regulatory_constraints,
        initial_policy: np.ndarray, # should be in a cellular encoding
        reward_function: np.ndarray, #### does not matter
        seed=0,
    ):
        
        def reward_function(flat_state, flat_action):
            return 0
    
        super().__init__(
            confidence_level=confidence_level,
            accuracy=accuracy,
            n_cells=n_cells,
            n_intracellular_states=n_intracellular_states,
            cellular_encoding=cellular_encoding,
            n_intracellular_actions=n_intracellular_actions,
            cellular_decoding=cellular_decoding,
            cell_classes=cell_classes,
            cell_labelling_function=cell_labelling_function,
            regulatory_constraints=regulatory_constraints,
            initial_policy=initial_policy,
            reward_function=reward_function,
            seed=seed,
        )
        self.reward_sum = np.zeros((self.n_states, self.n_actions), dtype = float)
        self.reward_estimates = np.zeros((self.n_states, self.n_actions), dtype = float)
        self.reward_errors = np.zeros((self.n_states, self.n_actions), dtype = float) 


    def update(self, state, reward, side_effects):
        self.reward_sum[self.flat_previous_state, self.flat_action] += reward
        super().update(state, reward, side_effects)

    def _extended_value_iteration(self):

        for flat_state in range(self.n_states):
            for flat_action in range(self.n_actions):
                self.reward_estimates[flat_state,flat_action] = self.reward_sum[flat_state, flat_action] / max([1, self.previous_episodes_count[flat_state, flat_action]])
                self.reward_errors[flat_state, flat_action] = np.sqrt(
                    (
                        7 * np.log(2 * self.n_states * self.n_actions * self.time_step / self.confidence_level) 
                    ) / (
                        2 * max(
                            [1,
                            self.previous_episodes_count[flat_state, flat_action]]
                        )
                    )
                )
                self.reward_function[flat_state, flat_action] = min(
                    [1,
                    self.reward_estimates[flat_state, flat_action] + self.reward_errors[flat_state, flat_action]]
                )
        return super()._extended_value_iteration()




class PeUcrlMinusShieldAgent(PeUcrlAgent):

    def _pe_shield(self):
        self.behaviour_policy = deepcopy(self.target_policy)


            

class PeUcrlMinusEviAgent(PeUcrlAgent):

    def _planner(self):
        
        self.target_policy = deepcopy(np.random.randint(0, self.n_intracellular_actions, size=(self.n_cells, self.n_states)))




class PeUcrlMinusRMinusShieldAgent(PeUcrlMinusRAgent):

    def _pe_shield(self):
        self.behaviour_policy = deepcopy(self.target_policy)




class PeUcrlMinusRMinusExperimentationAgent(PeUcrlMinusRAgent):

    def _extended_value_iteration(self):
        
        quality = np.zeros((self.n_states, self.n_actions))
        previous_value = np.zeros(self.n_states)
        current_value = np.zeros(self.n_states)
        stop = False
        while not stop:
            for flat_state in range(self.n_states):
                inner_max = [
                    np.sum(
                        [
                            self.transition_estimates[flat_state, flat_action, flat_next_state] * previous_value[flat_next_state] for flat_next_state in range(self.n_states)
                        ]
                    ) for flat_action in range(self.n_actions)
                ]
                quality[flat_state, :] = [(self.reward_estimates[flat_state, flat_action] + inner_max[flat_action]) for flat_action in range(self.n_actions)]
                quality[flat_state, :] *= 2 * self.transition_indicators[flat_state, :] - 1
                current_value[flat_state] = max(quality[flat_state, :])
                if current_value[flat_state] < 0:
                    print('there is an action-free state', flat_state)
                    current_value[flat_state] = 0
            diff = [current_value[flat_state] - previous_value[flat_state] for flat_state in range(self.n_states)]
            stop = (max(diff) - min(diff) < 1/self.time_step)
            previous_value = current_value
        self.value_function = current_value # for testing purposes
        return quality
    


    
class PeUcrlMinusSafetyAgent(PeUcrlMinusShieldAgent):

    def _action_pruning(self):
        return False
        


        
class PeUcrlMinusRMinusSafetyAgent(PeUcrlMinusRMinusShieldAgent):

    def _action_pruning(self):
        return False