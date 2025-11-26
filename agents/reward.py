HAND_REWARD_SCALE = 0.01

def calculate_reward(prev_hand_count, curr_hand_count, 
                     win=False, lose=False)->float:
    if win:
        return 1
    if lose:
        return -1
    
    if prev_hand_count < curr_hand_count:
        return HAND_REWARD_SCALE * (curr_hand_count - prev_hand_count)
    elif prev_hand_count > curr_hand_count:
        return HAND_REWARD_SCALE # since only way is to lose 1 card
    else:
        return 0

