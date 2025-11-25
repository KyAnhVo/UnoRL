# Agents to be used for UnoRL

## Divide by state representation

### Strat
State representation will be
```
Player hands:
    #red, #green, #blue, #yellow,
    #number, #skip, #reverse, #draw_2
    #wild, #wild_draw_4
Opponent:
    #card
Discarded deck:
    #red, #green, #blue, #yellow,
    #number, #skip, #reverse, #draw_2
    #wild, #wild_draw_4
Target card (top of played/discarded pile):
    top color onehot (index in order of
    red, green, blue, yellow)
    top suit onehot (index in order of 
    number, skip, reverse, draw_2, wild, wild_draw_4)
    top number onehot (index in order 0-9,
    if not a Number card then all 0)
```

### Card
State representation will be
```
Player hands:
    #card for each unique card (0, 1, or 2)
Opponent (in play order from player):
    #card
Discarded deck:
    #card for each unique card (0, 1, or 2)
Target card (top of played/discarded pile):
    top color onehot (index in order of
    red, green, blue, yellow)
    top suit onehot (index in order of 
    number, skip, reverse, draw_2, wild, wild_draw_4)
    top number onehot (index in order 0-9,
    if not a Number card then all 0)
```

### Tabular
Note: this is used for either SARSA, tabular Q, or
tabular OSL.
```
NEED THINKING
```

## Divide by model

### Deep Q
Off-policy algorithm, uses nn for high-dim state space.

### Deep MC
True return from "episode", pure on-policy, with true return
as label.

### Note
We have elected to not do any non-deep algorithm
due to the combinatorial explosion of state space. This essentially
leaves out the algorithms that we may have considered, which are
Certainty Equivalence (CE), State-Action-Reward-State-Action (SARSA),
and non-deep variances of the 2 policies named above.
