from agents.peucrl import PeUcrlAgent

class AntiSilencingPeUcrlAgent(PeUcrlAgent):

    pass

    silence = np.array(self.n_cells)

    def _side_effects_processing(
        self,
        next_state,
        side_effects,
    ):

        if self.time_step == 0: # change
            self.silence = np.zeros(self.n_cells)
        for reporting_cell in range(self.n_cells): # the same
            for reported_cell in range(self.n_cells):
                if self.side_effects[reporting_cell, reported_cell] == 'safe':
                    self.side_effects_functions[self.intracellular_states[reported_cell]] -= {'unsafe'}
                elif self.side_effects[reporting_cell, reported_cell] == 'unsafe':
                    self.side_effects_functions[self.intracellular_states[reported_cell]] -= {'safe'}
                else: # change
                    self.silence[reporting_cell] += 1


    def _cell_prioritisation(
        self,
        cell_set: set,
    ):

        if len(cell_set) == self.n_cells:
            self.last_cell = np.where(self.silence == self.silence.min())
            if len(self.last_cell) >= 2:
                self.last_cell = random.sample(self.last_cell, 1)
        if len(cell_set) >= 2:
            cell = random.sample(cell_set - self.last_cell)
        else:
            cell = self.last_cell
        return cell