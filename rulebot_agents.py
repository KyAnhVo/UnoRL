from agents.deepmc_card     import DeepMCCardAgent
from agents.deepmc_strat    import DeepMCStratAgent
from agents.deepq_card      import DeepQCardAgent
from agents.deepq_strat     import DeepQStratAgent

class DeepMCCardRulebot(DeepMCCardAgent):
    def __init__(self):
        super().__init__()
        self.MODEL_HISTORY_DIR = 'rulebot_model_history'

class DeepMCStratRulebot(DeepMCStratAgent):
    def __init__(self):
        super().__init__()
        self.MODEL_HISTORY_DIR = 'rulebot_model_history'

class DeepQCardRulebot(DeepQCardAgent):
    def __init__(self):
        super().__init__()
        self.MODEL_HISTORY_DIR = 'rulebot_model_history'

class DeepQStratRulebot(DeepQStratAgent):
    def __init__(self):
        super().__init__()
        self.MODEL_HISTORY_DIR = 'rulebot_model_history'

