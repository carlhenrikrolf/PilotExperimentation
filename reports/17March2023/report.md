# Ablation Studies
- In this document, we report the results from ablation studies on our procedure PE-UCRL.
  - First, we provide some background by describing the environment. We also describe the constraints the PE-UCRL should satisfy.
  - Then, we report results from the ablation studies.
    - First: safety, second: convergence, third: time complexities.
    - Either the shield is removed from the algorithm (PE-UCRL\Shield) or action pruning is removed (PE-UCRL\ActionPruning) or both (PE-UCRL\ActionPruning\Shield).
  - Finally, we provide some next steps.
- The environment is a simple model of polarisation in news recommender engines.
  - The recommender engine itself the agent and the actions of the agent are the recommendations.
  - There are 2 users[^-1].
  - One of them also works as a moderator, who can observe both the recommendations they receive and the recommendations that other users receive.
  - The users can be polarised on a spectrum from left to right. There are 4 degrees of polarisation.
  - The moderator considers the leftmost degree and the rightmost degree to be unsafe.
  - There are 4 recommendations. Each drives the user to a certain degree of polarisation.
  - The most extreme degrees of polarisation tends to give the highest reward. So, reward is anti-correlated with safety.
- The regulatory constraint is ```P>=0.8 [ G n <= 1]``` (as expressed in the PRISM language), where ```n``` is a formula describing the number of cells that experiences a side effect.
  - In other words, at any point, at most half the users may be extremely polarised.
  - Note that convergence cannot be guaranteed under this constraint. The reason is that we use ```G``` instead of ```F<=```, which is currently not allowed for parametric model-checking in PRISM.



## Safety

- Table 1 shows the results of ablation with respect to safety.
    - The top left table cell (with action pruning and shield) shows results for PE-UCRL, whereas the bottom right table cell (without action pruning and shield) shows an adaptation of UCRL2 to cellular MDPs.
    - Each table cell shows the results from two seeds for the agent.
    - The plots show the incidence of side effects as a function of time. A kind of rollling average but where average has been replaced by max is used to make the plot more easily read.
    - Below the plots, the max of the side effects incidence is indicated in text.

**Table 1.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed* ![](../../results/.juju_peucrl_2/rolling_max_side_effects_incidence_vs_time.png) max: 50% | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/rolling_max_side_effects_incidence_vs_time.png) max: 50%
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/rolling_max_side_effects_incidence_vs_time.png) max: 50% <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/rolling_max_side_effects_incidence_vs_time.png) max: 100% | *1st seed* ![](../../results/.juju_peucrl_minus_safety_1/rolling_max_side_effects_incidence_vs_time.png) max: 100% <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/rolling_max_side_effects_incidence_vs_time.png) max: 100%

- The results are as expected.
- PE-UCRL has a maximum side effects incidence of 50 %, i.e., 1 user can be extremely polarised.
- That is also true for PE-UCRL\ActionPruning. The difference is that, in PE-UCRL, the side effects incidence decreases over time (very quickly).
- In PE-UCRL\Shield, the side effects incidence also drops quickly, however, there is no guarantee that it is bounded above at all time steps. It may happen by accident (1st seed) since unsafe actions are removed quickly, but there is no guarantee (see 2nd seed).
- In PE-UCRL\ActionPruning\Shield, the side effects incidence is maximised since it is correlated with reward in our polarisation environment.

## Convergence

- Table 2 shows the results of ablation with respect to convergence.
    - The table is structured in a similar way as in Table 1.
    - In this case, it is a normal rolling average.

**Table 2.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_2/rolling_mean_reward_vs_time.png) | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/rolling_mean_reward_vs_time.png) <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/rolling_mean_reward_vs_time.png)
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/rolling_mean_reward_vs_time.png) | *1st seed* ![](../../results/.juju_peucrl_minus_safety_1/rolling_mean_reward_vs_time.png) <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/rolling_mean_reward_vs_time.png)

- The results show that policies tend to improve. However, they do not show that very conclusively. This may have to do with too few time steps. It may also be because the requirements for convergence are not satisfied by the environment---as they would extremely rarely be satisfied in real-world applications.
  - The noise in the beginning comes from the fact that the agent has not yet learned much and is exploring randomly.
  - The noisy periods are replaced by oscillations of higher and lower reward. The highs of these oscillations are better than the initial policy. This indicates that the agent is in a period of exploration by optimism in the face of uncertainty.
  - PE-UCRL\ActionPruning\Shield receives considerably more reward, but the cost is maximal side effects incidence.
  - The theory suggests that eventually the oscillations should end and the reward should stabilise around a high value, but there may be too few time steps here to see that.
  - 3 out of 8 plots do look peculiar, and apart from increasing the number of time steps, increasing the number of seeds could give more reliable results. Maybe these agents were simply initialised with unfortunate choices of the seeds such that they learn slowly in the beginning.
  - One assumption that needs to be satisfied for optimality pertains to the implementation of the algorithm. In the current implementation, some states are transient (they have no transitions), but that poses problems for convergence.

## Time Complexity

- Table 3 shows the results of ablation with respect to time complexities.
    - The table is structured in a similar way as in tables 1 and 2.
    - The plots show when the agents perform an update. The agent may perform an update because of the new episode criterion from UCRL2 or because the agent just observed an unsafe transition. The update includes planning the target policy using extended value iteration and then applying a shield that uses model-checking to safely merge the behaviour policy with the target policy.
    - Associated with most of the plot are the following data: the computation time between time steps (mean and standard deviation; the number of samples is equal to the number of time steps, i.e. $10^6$), the computation for an update ("between episodes"), and the number of episode updates (this is the number of samples for the mean and the standard deviation of "between episodes").


**Table 3.**
![]() | with action pruning | without action pruning
---|---|---
**with shield** | *1st seed* ![](../../results/.juju_peucrl_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (426 ± 107) ms <br> number of episodes: 34<br> *2nd seed* ![](../../results/.juju_peucrl_2/episodes.png) | *1st seed* ![](../../results/.juju_peucrl_minus_action_pruning_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (423 ± 48) ms <br> number of episodes: 81 <br> *2nd seed*[^0] ![](../../results/.juju_peucrl_minus_action_pruning_2/episodes.png)
**without shield** | *1st seed* ![](../../results/.juju_peucrl_minus_shield_1/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (470 ± 71) ms <br> number of episodes: 71 <br> *2nd seed* ![](../../results/.juju_peucrl_minus_shield_2/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (539 ± 86) ms <br> number of episodes: 71 | *1st seed*[^1] ![](../../results/.juju_peucrl_minus_safety_1/episodes.png) between time steps: (25 ± 680) ms <br> between episodes: (4298 ± 56986) ms <br> number of updates: 318 <br> *2nd seed* ![](../../results/.juju_peucrl_minus_safety_2/episodes.png) between time steps: (11 ± 1) ms <br> between episodes: (561 ± 75) ms <br> number of episodes: 346

- The results are mostly as expected and nothing is inconsistent with the theory.
  - As expected the updates happen often in the beginnign and then less and less.
  - As expected the computation time between time steps is much shorter than between episodes.
  - As expected, increased safety---either through action pruning or shielding---makes learning slower. That is, there are fewer updates. For shielding, fewer policies can be experimented with within any given time. For action  pruning there are fewer policies to explore.
   - Unexpectedly, the computation time of the shielding does not increase the overall computation time of the update. This could be explained by planning through extended value iteration being much more expensive. Especially as more iterations have to be performed as time progresses.

## Next steps
- [ ] improve experimental pipeline
    - [ ] parallelise
- [ ] extend prism implementation to more suitible constraints using the next operator
- [ ] implement comparisons
    - [ ] AlwaysSafe
    - [ ] DOPE
    - [ ] PSO, I have asked for pseudocode as I don't find the AAAI paper really reproducible

## Footnotes

[^-1]: Note that this is not the only way to parameterise the environment.

[^0]: The process died during the run, so it is incomplete.

[^1]: The 1st seed without shield and action pruning has much larger values, this is probably because it was run with the debugger attached. Therefore, the 1st seed without action pruning and shield can be ignored for time complexity analysis.