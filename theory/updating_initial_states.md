Consider two shielding methods: (without updating) $\mathbf s^\mathrm{init}, \mathscr M[k] \models \varPhi$ and (with updating) $\mathbf s[k], \mathscr M[k] \models \varPhi$.

**Proposition.** 
Assume: $\varPhi$ deals with determinstic constraints, but they are wrapped in a model $\mathscr M[k]$ with uncertainty $1-\delta$.
Then: If the agent ends up in a state violating $\varPhi$, the (with updating) method, may be able to recover a policy satisfying $\varPhi$ again.
The (without updating) method will never update its policy again.

_Proof._
The scenario is possible with probability bounded by $1-\delta$.
For the (without updating) method, all policies going through that state would be unsafe.

Intuitively, (with updating) is only better in the best case scenario, it does not improve the guarantees.