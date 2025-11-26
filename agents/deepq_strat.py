from agents.state_translator import STRAT_STATE_DIM_COUNT, strategic_state_translate
from agents.deep_uno_agent import DeepUnoAgent
from agents.deeprl_nn import DeepRL_NN

from typing import override, List
import torch

class DeepQStratAgent(DeepUnoAgent):
    target_nn: DeepRL_NN
    episodes: int

    def __init__(self):
        super().__init__(state_dim=STRAT_STATE_DIM_COUNT)
        self.target_nn = DeepRL_NN(state_dim=STRAT_STATE_DIM_COUNT,
                                   action_dim=61)
        self.target_nn.load_state_dict(self.online_nn.state_dict())
    
    @override
    def state_translation(self, state) -> List[int]:
        return strategic_state_translate(state)

    @override
    def compute_targets(self) -> List[float]:
        device = self.target_nn.device

        with torch.no_grad():
            next_states = torch.tensor(
                self.next_state_list,
                dtype=torch.float32,
                device=device,
            )

            rewards = torch.tensor(
                self.rewards_list,
                dtype=torch.float32,
                device=device,
            )

            dones = torch.tensor(
                self.dones,
                dtype=torch.float32,
                device=device,
            )

            # Q_target(s', ·)
            next_q_values = self.target_nn(next_states)
            max_next_q = next_q_values.max(dim=1).values

            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        return targets.cpu().tolist()




