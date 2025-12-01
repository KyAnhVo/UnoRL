from agents.deepq import DeepQAgent
from agents.state_translator import CARD_STATE_DIM_COUNT, card_state_translate, card_state_reward
from typing import override, List

class DeepQCardAgent(DeepQAgent):
    def __init__(self):
        super().__init__(state_dim=CARD_STATE_DIM_COUNT)
        self.FILE_NAME = "qcard"
    
    @override
    def state_translation(self, state) -> List[int]:
        return card_state_translate(state)
    
    @override
    def calculate_reward(self, prev_state: List[int], curr_state: List[int]) -> float:
        return card_state_reward(
                prev_state=prev_state,
                curr_state=curr_state,
                gain_card_penalty=self.GAIN_CARD_PENALTY,
                lose_card_reward=self.LOSE_CARD_REWARD
                )

