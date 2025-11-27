from typing import Dict, Tuple, List
from enum import IntEnum

'''
Game state given by RLCard env

NOTE: Card is denoted by "color-number" where
- color:    [r, g, b, y]
- number:   [0-9] + [reverse, skip, draw_2, wild, wild_draw_4]

{
    "obs": <trash>,
    "legal_actions": [
        list of legal action ids
    ],
    "raw_legal_actions": [
        list of legal card plays (or draw)
    ],
    
    # actual useful things to do state translation
    "raw_obs": {
        "hand": [
            list of cards on hand
        ],
        "target": card most recently played,
        "played_cards": [
            list of played cards
        ],
        "others_hand": [
            list of cards on other's hand (DO NOT USE)
        ],
        "legal_actions": [
            list of playable cards, or "draw" if
            no card is playable
        ],
        "card_num": [
            hand count of each player, by player ids
        ],
        "player_num": 2,
        "current_player": player id,
    },

    # might be useful?
    "action_record": [
        list of actions, each item is a list:
        [player id, played card]
    ]
}
'''
# hand: 10, op: 1, discarded: 10, top: 20
STRAT_STATE_DIM_COUNT = 41

# hand: 60, op: 1, discarded: 60, top: 20
CARD_STATE_DIM_COUNT = 141

class Color(IntEnum):
    R = 0
    G = 1
    B = 2
    Y = 3

class Suit(IntEnum):
    NUMBER = 0
    SKIP = 1
    REVERSE = 2
    DRAW_2 = 3
    WILD = 4
    WILD_DRAW_4 = 5

def card_to_int(card: str)->int:
    val = 0
    card_detached = card.split('-')
    color_str, value_str = card_detached[0], card_detached[1]

    match color_str:
        case 'r':
            val += Color.R
        case 'g':
            val += Color.G
        case 'b':
            val += Color.B
        case 'y':
            val += Color.Y
    
    val *= 15

    match value_str:
        case v if v.isdigit():
            val += int(v)
        case 'skip':
            val += Suit.SKIP + 9
        case 'reverse':
            val += Suit.REVERSE + 9
        case 'draw_2':
            val += Suit.DRAW_2 + 9
        case 'wild':
            val += Suit.WILD + 9
        case 'wild_draw_4':
            val += Suit.WILD_DRAW_4 + 9

    return val

def int_to_action(action_int: int)->str:
    if action_int == 60:
        return 'draw'
    
    color = action_int // 15
    suit = action_int % 15
    ret = ''
    match color:
        case Color.R:
            ret += 'r-'
        case Color.G:
            ret += 'g-'
        case Color.B:
            ret += 'b-'
        case Color.Y:
            ret += 'y-'
    
    if 0 <= suit <= 9:
        ret += str(suit)
    else:
        suit -= 9
        match suit:
            case Suit.SKIP:
                ret += 'skip'
            case Suit.REVERSE:
                ret += 'reverse'
            case Suit.DRAW_2:
                ret += 'draw_2'
            case Suit.WILD:
                ret += 'wild'
            case Suit.WILD_DRAW_4:
                ret += 'wild_draw_4'

    return ret


def translate_card(card: str)->Tuple[Color, Suit, int]:
    card_detached = card.split('-')
    color_str, value_str = card_detached[0], card_detached[1]
    
    color: Color = Color.R
    match color_str:
        case 'r':
            color = Color.R
        case 'g':
            color = Color.G
        case 'b':
            color = Color.B
        case 'y':
            color = Color.Y

    suit: Suit = Suit.NUMBER
    num: int = -1
    match value_str:
        case v if v.isdigit():
            suit = Suit.NUMBER
            num = int(v)
        case 'skip':
            suit = Suit.SKIP
        case 'reverse':
            suit = Suit.REVERSE
        case 'draw_2':
            suit = Suit.DRAW_2
        case 'wild':
            suit = Suit.WILD
        case 'wild_draw_4':
            suit = Suit.WILD_DRAW_4

    return color, suit, num

# TODO: implement state translation for tabular
def tabular_state_translate(state: Dict):
    pass

def strategic_state_translate(state: Dict)->List[int]:
    ''' Translate from game given state to strat state

    Player hands: [0-9]
    #red, #green, #blue, #yellow
    #number, #skip, #reverse, #draw_2
    #wild, #wild_draw_4

    Opponent: [10-10]
    #card

    Discarded deck: [11-20]
    #red, #green, #blue, #yellow,
    #number, #skip, #reverse, #draw_2
    #wild, #wild_draw_4

    Target card (top of played/discarded pile): [21-40]
    top color onehot (index in order of
    red, green, blue, yellow) [21-24]
    top suit onehot (index in order of 
    number, skip, reverse, draw_2, wild, 
    wild_draw_4) [25-30]
    top number onehot (index in order 0-9,
    nothing is 1 if suit is not Number) [31-40]
    '''

    real_state = state['raw_obs']
    strat_state = [0 for _ in range(41)]

    agent_player_id = real_state['current_player']
    opp_player_id = 0 if agent_player_id == 1 else 1

    # agent hand
    color_start_index = 0
    suit_start_index = 4
    for card in real_state['hand']:
        color, suit, _ = translate_card(card)
        if suit != Suit.WILD and suit != Suit.WILD_DRAW_4:
            strat_state[color_start_index + color] += 1
        strat_state[suit_start_index + suit] += 1

    # opponent hand count
    strat_state[10] = real_state['card_num'][opp_player_id]

    # discarded deck
    color_start_index = 11
    suit_start_index = 15
    for card in real_state['played_cards']:
        color, suit, _ = translate_card(card)
        if suit != Suit.WILD and suit != Suit.WILD_DRAW_4:
            strat_state[color_start_index + color] += 1
        strat_state[suit_start_index + suit] += 1

    # target card
    color_start_index = 21
    suit_start_index = 25
    number_start_index = 31
    color, suit, number = translate_card(real_state['target'])
    strat_state[color_start_index + color] = 1
    strat_state[suit_start_index + suit] = 1
    # NOTE: For safety
    if suit == Suit.NUMBER and number != -1:
        strat_state[number_start_index + number] = 1
    
    return strat_state

def card_state_translate(state: Dict):
    ''' Translate from env given state to card state:
    '''
    real_state = state['raw_obs']
    card_state = [0 for _ in range(CARD_STATE_DIM_COUNT)]

    agent_player_id = real_state['current_player']
    opp_player_id = 0 if agent_player_id == 1 else 1

    # agent hand
    start_index = 0
    for card in real_state['hand']:
        index = 0
        color, suit, number = translate_card(card)
        index += color * 15
        if number != -1 and suit != Suit.NUMBER:
            index += number
        else:
            index += suit + 9
        card_state[start_index + index] += 1

    # opp hand count
    card_state[60] = real_state['card_num'][opp_player_id]

    # discarded deck
    start_index = 61
    for card in real_state['played_cards']:
        index = 0
        color, suit, number = translate_card(card)
        index += color * 15
        if number != -1 and suit != Suit.NUMBER:
            index += number
        else:
            index += suit + 9
        card_state[start_index + index] += 1

    # target card
    color_start_index = 121
    suit_start_index = 125
    number_start_index = 131
    color, suit, number = translate_card(real_state['target'])
    card_state[color_start_index + color] = 1
    card_state[suit_start_index + suit] = 1
    if suit == Suit.NUMBER and number != -1:
        card_state[number_start_index + number] = 1

    return card_state

def strat_state_reward(
        prev_state: List[int], 
        curr_state: List[int], 
        gain_card_penalty: float, 
        lose_card_reward: float)->float:
    initial_state = True
    for i in (prev_state):
        if i != 0:
            initial_state = False
            break

    if initial_state:
        return 0

    card_count_1 = sum(prev_state[4:10])
    card_count_2 = sum(curr_state[4:10])

    card_change = card_count_2 - card_count_1
    if card_change <= 0:
        return -lose_card_reward * card_change
    else:
        return -gain_card_penalty * card_change

def card_state_reward(
        prev_state: List[int], curr_state: List[int],
        gain_card_penalty: float,
        lose_card_reward: float)->float:
    initial_state = True
    for i in prev_state:
        if i != 0:
            initial_state = False
            break
    if initial_state:
        return 0

    card_count_1 = sum(prev_state[0:60])
    card_count_2 = sum(curr_state[0:60])
    card_change = card_count_2 - card_count_1
    if card_change <= 0:
        return -lose_card_reward * card_change
    else:
        return -gain_card_penalty * card_change
