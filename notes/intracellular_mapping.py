from email.headerregistry import HeaderRegistry
import environment as env
n_cells = env.n_cells

def intracellular_mapping(
    state=None,
    action=None,
):
    assert state != None ^ action != None # xor

    # insert mapping here

    if state != None:
        return #integer list of length n_cells
    else: # if action != None
        return #integer list of length n_cells
