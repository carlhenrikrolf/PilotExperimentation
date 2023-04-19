# Constraints

## Benchmarks

__PeUcrl__

- ```P>=0.01 [ F<=1 n <= 1 ]```

__PSO__

```pm
label "phi0" = G n_nonconsenting <= 0;
label "phi1" = G n_children <= 0;
P>=1 [ "phi0" & "phi1" ];
```

__AlwaysSafe__

__DOPE__

Model file:
```pm
rewards "cost"
true : n_nonconsenting + n_children;
endrewards
```
Properties file:
```pm
R{"cost"}<=1 [ C <= 5 ];
```
- [ ] Check if there is indeed a horizon for DOPE

__Temporal-logic based methods__

```pm
label "unsafe" = n > 0;
label "unsafe_child" = n_children > 0;
P<=0.95 [ G ! "unsafe_child" & F ! "unsafe" ]
```

__Impact-related__

Model file
```pm
rewards "aux0"
s = 0 : 0.3;
s = 1 : 0.1;
endrewards

rewards "aux1"
s = 0 : 0.9;
s = 1 : 0.2;
endrewards
```

Properties file:
```pm
R{"aux0"}>=0.01 [ C <= t ] / t & R{"aux1"}>=0.01 [ C <= t ] / t;
```

- [ ] Need to check that this holds
- [ ] Need to check how to reduce this into 1 property

## Applications

Clinical Trials

Ecotoxicity Testing

GDPR recommender engines

- ```P>=0.99 [ G n_nonconsenting <= 0] & P>=0.2 [ G n <= 1]```

## Speed-Related

This seems like future work.
Maybe, I can comment on it at the high level?

- $D_{KL}(\pi^{t+1} \| \pi^t) \leq ub$

