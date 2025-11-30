from agents.deepq import DeepQAgent
from agents.state_translator import STRAT_STATE_DIM_COUNT, strategic_state_translate, strat_state_reward
from typing import override, List
import torch

class DeepQStratAgent(DeepQAgent):
    def __init__(self):
        super().__init__(state_dim=STRAT_STATE_DIM_COUNT)
        self.FILE_NAME = "qstrat"
    
    @override
    def state_translation(self, state) -> List[int]:
        return strategic_state_translate(state)
    
    @override
    def calculate_reward(self, prev_state: List[int], curr_state: List[int]) -> float:
        return strat_state_reward(
                prev_state=prev_state,
                curr_state=curr_state,
                gain_card_penalty=self.GAIN_CARD_PENALTY,
                lose_card_reward=self.LOSE_CARD_REWARD
                )
    
    @override
    def after_game(self, payoff: int):
        # Call parent's after_game
        super().after_game(payoff)
        

