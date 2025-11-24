# Agents to be used for UnoRL

## Divide by state representation

### Strat
State representation will be
```
Player hands:
    #red, #blue, #yellow, #green,
    #numbers, #actions,
    #wilds, #wild+4s
Opponent (in play order from player):
    opp1 #card, opp2 #card, opp3 #card
Discarded deck:
    #red, #blue, #yellow, #green,
    #numbers, #actions,
    #wilds, #wild+4s
```

### Card
State representation will be
```
Player hands:
    #card for each unique card (0, 1, or 2)
Opponent (in play order from player):
    opp1 #card, opp2 #card, opp3 #card
Discarded deck:
    #card for each unique card (0, 1, or 2)
```

## Divide by model

### Deep Q
Off-policy algorithm, uses nn for high-dim state space.

### Deep MC
True return from "episode", pure on-policy, with true return
as label.

### Note
We have elected to not do any non-deep state space
due to the combinatorial explosion of state space. This essentially
leaves out the algorithms that we may have considered, which are
Certainty Equivalence (CE), State-Action-Reward-State-Action (SARSA),
and non-deep variances of the 2 policies named above.
