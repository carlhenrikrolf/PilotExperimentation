# Ablation Studies
- In this document, we report the results from ablation studies on our procedure PE-UCRL.
  - First, we provide some background by describing the environment. We also describe the constraints the PE-UCRL should satisfy.
  - Then, we report results from the ablation studies.
    - First: safety, second: convergence, third: time complexities.
    - Either the shield is removed from the algorithm (PE-UCRL\Shield) or action pruning is removed (PE-UCRL\ActionPruning) or both (PE-UCRL\ActionPruning\Shield).
  - Finally, we provide some conclusions and next steps.
- The environment is a simple model of polarisation in news recommender engines.
  - The recommender engine itself the agent and the actions of the agent are the recommendations.
  - There are 2 users[^-1].
  - One of them also works as a moderator, who can observe both the recommendations they receive and the recommendations that other users receive.
  - The users can be polarised on a spectrum from left to right. There are 4 degrees of polarisation.
  - The moderator considers the leftmost degree and the rightmost degree to be unsafe.
  - There are 4 recommendations. Each drives the user to a certain degree of polarisation.
  - The most extreme degrees of polarisation tends to give the highest reward. So, reward is anti-correlated with safety.
- The regulatory constraint is ```P>=0.8 [ G n <= 1]``` (as expressed in the PRISM language), where ```n``` is a formula describing the number of cells that experiences a side effect.
    - note convergence may not be permissible under this choice, but current prism implementation does not allow for nontrivial permissible constraints.



## Safety

- Table 1 shows the results of ablation with respect to safety.
    - The top left table cell (with action pruning and shield) shows results for PE-UCRL, whereas the bottom right table cell (without action pruning and shield) shows an adaptation of UCRL2 to cellular MDPs.
    - Each table cell shows the results fr two seeds for the agent.

**Table 1.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed* ![](../../results/.juju_peucrl_2/rolling_max_side_effects_incidence_vs_time.png) max: 50% | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/rolling_max_side_effects_incidence_vs_time.png) max: 50%
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/rolling_max_side_effects_incidence_vs_time.png) max: 100% | *1st seed* ![](../../results/.juju_peucrl_minus_safety_1/rolling_max_side_effects_incidence_vs_time.png) max: 100% <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/rolling_max_side_effects_incidence_vs_time.png) max: 100%


## Convergence

- Table 2 shows the results of ablation with respect to convergence.
    - The table is structured in a similar way as in Table 1.

**Table 2.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_2/rolling_mean_reward_vs_time.png) | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/rolling_mean_reward_vs_time.png) <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/rolling_mean_reward_vs_time.png)
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/rolling_mean_reward_vs_time.png) | *1st seed* ![](../../results/.juju_peucrl_minus_safety_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/rolling_mean_reward_vs_time.png)

## Time Complexity

- Table 3 shows the results of ablation with respect to time complexities.
    - The table is structured in a similar way as in Table 1 and 2.

**Table 3.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (426 ± 107) ms <br> number of episodes: 34<br> *2nd seed* ![](../../results/.juju_peucrl_2/episodes.png) | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (423 ± 48) ms <br> number of episodes: 81 <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/episodes.png)
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (470 ± 71) ms <br> number of episodes: 71 <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (539 ± 86) ms <br> number of episodes: 71 | *1st seed*[^1] ![](../../results/.juju_peucrl_minus_safety_1/episodes.png) between time steps: (25 ± 680) ms <br> between episodes: (4298 ± 56986) ms <br> number of updates: 318 <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (561 ± 75) ms <br> number of episodes: 346

- Other observations:
    - 

## Conclusions
- For PE-UCRL with neither shield nor action pruning, convergence should be clearly illustrated.
    - Probably, there are not enough time steps. Other work has used $10^8$ time steps rather than $10^6$.
    - Running more agent seeds
    - A more subtle problem has to do with transient states.

### Next steps
- [ ] improve experimental pipeline
    - [ ] parallelise
    - [ ] save agent and state of environment
- [ ] extend prism implementation to more suitible constraints using the next operator
- [ ] implement comparisons
    - [ ] AlwaysSafe
    - [ ] DOPE
    - [ ] PSO, I have asked for pseudocode as I don't find the AAAI paper really reproducible

## Footnotes

[^-1]: Note that this is not the only way to parameterise the environment.

[^0]: The process died during the run, so it is incomplete.

[^1]: The 1st seed without shield and action pruning has much larger values, this is probably because it was run with the debugger attached. Therefore, the 1st seed without action pruning and shield can be ignored for time complexity analysis.