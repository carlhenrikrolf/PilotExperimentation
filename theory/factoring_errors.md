Define
$$\mathit{EB}(\mathbf s, \mathbf a) = \sqrt{\frac{14|\mathscr S|\log(2|\mathscr A|t[k]/\delta)}{\max\{1, \tilde N(\mathbf s, \mathbf a)\}}}$$
and
$$\mathit{eb}_i(s_i, a_i) = \sqrt{\frac{14|\mathscr S_i|\log(2|\mathscr A_\#|t[k]/\delta)}{\max\{1, N_\#(s_i, a_i)\}}}$$

**Proposition.** Assume: $\mathit{eb}_i \geq x$ and $\mathit{EB} \geq y$.
Then: $x \leq y$.

_Proof._ This is true since $|\mathscr S_i|\leq|\mathscr S|$, $|\mathscr A_\#|\leq|\mathscr A|$ and $\tilde N(\mathbf s, \mathbf a) \geq N_\#(s_i, a_i)$.

Intuitively, this result says that in the best case, using factored error bounds converges more quickly to a better policy. However, we have no proof that this is always the case.