from agents.deep_uno_agent import DeepUnoAgent
from agents.deeprl_nn import DeepRL_NN
from typing import override, List
import torch

class DeepQAgent(DeepUnoAgent):
    target_nn: DeepRL_NN

    # sync target_nn param using online_nn after
    # every SYNC_RATE episodes
    SYNC_RATE: int


    def __init__(self, state_dim: int):
        super().__init__(state_dim=state_dim)
        self.target_nn = DeepRL_NN(state_dim=state_dim, action_dim=61)
        self.target_nn.load_state_dict(self.online_nn.state_dict())

        # lower == more unstable model, but train more frequent
        self.SYNC_RATE = 500


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
            next_q_values = self.target_nn(next_states)
            max_next_q = next_q_values.max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        return targets.cpu().tolist()

    def after_game(self, payoff: int):
        super().after_game(payoff)
        # Sync target network periodically
        if self.episode_count % self.SYNC_RATE == self.SYNC_RATE - 1:
            self.target_nn.load_state_dict(self.online_nn.state_dict())

