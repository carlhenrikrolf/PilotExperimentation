# Comparisons

## Ablations

- Ablated PeShield
- Ablated ActionPruning
- Both of them ablated, i.e., an unsafe baseline

## Safe Exploration Procedures

- AlwaysSafe/PSO (variables ~ cells)
  - When adapted to a cellular MDP where the variables correspond to cells, the cellular AlwaysSafe and the cellular PSO become basically inseparable.
    - PSO works by labelling certain cells as _delicate_. The procedure is constrained to not influence these cells by removing edges from the CID. In the cellular model, this corresponds to removing intracelular action $a_i$ if cell $i$ is delicate.
    - AlwaysSafe works by assuming knowledge of the side effects function. Furthermore, the side effects function is assumed to be irrelevant for the constraints for some of the cells. Furthermore, the dynamics for those cells are also assumed to be known. This is contrary to the cellular MDP model. Therefore, we adapt ALwaysSafe to assume that the worst case dynamics and side effects labels are known. This means that we are basically labelling cells as delicate, and therefore, it is basically the same as PSO.
  - The input here could be a set containing all the cells that can be ingored in terms of safety (i.e. the non-delicate cells), e.g., ```['children', 'unemployed']```.
- eGreedy
  - Meant to be modelling when experiments are not coordinated, but reporting of side effects is coordinated.
    - Better modelled as eOptimistic
  - The input could for example be ```lambda t: 1/max(1,t)```
    - maybe simplify to a constant here
- Bounded Divergence
  - Used naively for some policy gradient algorithms, not even specifically for safety.
  - The input could look something like ```{'KL': 0.4, 'SED': 0.2}```
    - KL might not be that good, see table below of determinstic p,q in D(p||q). It is assumed that 0*(-inf)=0.

q | p=0 | p=1
---|---|---
=0 | 0 | inf
=1 | 0 | 1

- Alex Turner stuff, AUP (attainable utility preservation)
  - I guess some kind of weight for how much to put into the random reward functions should go here

## Left Out

- AlwaysSafe/PSO (variables ~ intracellular variables)
  - A more naturual way to implement AlwaysSafe/PSO would be to let each cell contain a number of intracellular variables (that can possibly be repeated over different cells). (Expressing the entire model as a CID, the model would consist of n or more connected components that are not connected to one another.)
  - This would be more complementary rather than comparable. For comparisons, PE-UCRL vs PE-UCRL+AlwaysSafe would make more sense for example.
  - The input here might have to be something on the form ```R{"..."}=? [ C ] + R{"..."}=? [ C ] <= 7``` for AlwaysSafe.
- Ergodic methods
  - For the same reason we leave these out
  - They would work well with the scrambling for what cells to explore.
- Bounded Lipschitz
  - PE-UCRL is a kind of extension of similar ideas to a different setting
    - Does not assume continuity (or other kinds of ordering in the state space)
  - The problem with using Lipschitz-continuity naively is that it relies on norms
    - For the cellular MDP model, we would like to use that $\| T(s,a) - T(s,b) \| \leq L \| a - b \|, L = 1$.
    - Trying to define $\|\cdot\|$ in that way while keeping it a norm is difficult though. For example, $\| x \| \coloneqq \sum_i \mathbb{I}[x_i \neq 0]$ is not a norm.
    - $\| x \| \coloneqq \sum_i |x_i|$ is a norm, but the bound would have to be $\| T(s,a) - T(s,b) \| \leq L \| a - b \|, L = |S_\#|$ which is very loose.