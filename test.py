state = [-1]
next_state = []

passed_state = [0, 1, 2, 3, 4]

for i in passed_state:
    state.append(i)
    next_state.append(i)

next_state.append(-1)

print(state)
print(next_state)
