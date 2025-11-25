from agents.state_translator import STRAT_STATE_DIM_COUNT
from agents.deep_uno_agent import DeepUnoAgent
from agents.deeprl_nn import DeepRL_NN

class DeepQStratAgent(DeepUnoAgent):
    target_nn: DeepRL_NN
    def __init__(self):
        super().__init__(state_dim=STRAT_STATE_DIM_COUNT)
        self.target_nn = DeepRL_NN(state_dim=STRAT_STATE_DIM_COUNT,
                                   action_dim=61)
    

