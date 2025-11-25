from agents.state_translator import STRAT_STATE_DIM_COUNT
from agents.deep_uno_agent import DeepUnoAgent

class DeepQStratAgent(DeepUnoAgent):
    def __init__(self):
        super().__init__(state_dim=STRAT_STATE_DIM_COUNT)

