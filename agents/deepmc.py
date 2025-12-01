from typing import override
from agents.deep_uno_agent import DeepUnoAgent
from typing import List

class DeepMCAgent(DeepUnoAgent):
    def __init__(self, state_dim: int):
        super().__init__(state_dim)

    @override
    def compute_targets(self) -> List[float]:
        state_count = len(self.state_list)
        target_lst = [0.0] * state_count
        G = 0.0
        for i in reversed(range(state_count)):
            if self.dones[i]:
                G = 0
            G = self.rewards_list[i] + self.gamma * G
            target_lst[i] = G
        return target_lst
