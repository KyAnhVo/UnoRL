from abc import ABC, abstractmethod
from typing import List
from agents.deeprl_nn import DeepRL_NN

class DeepUnoAgent(ABC):
    state_dim: int
    gamma: float

    online_nn: DeepRL_NN

    state_list:         List[List[int]]
    next_state_list:    List[List[int]]
    action_list:        List[int]
    rewards_list:       List[float]
    dones:              List[bool]

    def __init__(self, state_dim: int, gamma: float = 0.99):
        self.state_dim = state_dim
        self.gamma     = gamma

        # networks (only DQN-style subclasses will add target_nn)
        self.online_nn = DeepRL_NN(state_dim=state_dim, action_dim=61)

        # episode buffers
        self.state_list      = []
        self.next_state_list = []
        self.action_list     = []
        self.rewards_list    = []
        self.dones           = []

    # ------------------------------------------------------
    # RLCard-required API
    # ------------------------------------------------------

    @abstractmethod
    def step(self, state) -> int:
        """Action selection during training (epsilon-greedy)."""
        pass

    @abstractmethod
    def eval_step(self, state) -> int:
        """Action selection during evaluation (greedy)."""
        pass

    def use_raw(self) -> bool:
        """'False' means expect processed env states."""
        return False

    # ------------------------------------------------------
    # Required for your architecture
    # ------------------------------------------------------

    @abstractmethod
    def state_translation(self, state)->List[int]:
        """Convert RLCard state to encoded vector."""
        pass

    @abstractmethod
    def compute_targets(self) -> List[float]:
        """Compute learning targets (MC returns, TD targets, etc.)."""
        pass

    def train_online_nn(self):
        """Generic training hook: compute targets then train network."""
        targets = self.compute_targets()
        self.online_nn.train_batch(
            state_list      = self.state_list,
            actions_taken   = self.action_list,
            real_values     = targets,
        )

    # ------------------------------------------------------
    # Helpers that SHOULD be included
    # ------------------------------------------------------

    def record_transition(self, state, action, reward, next_state, done):
        """Add a transition to buffers."""
        self.state_list.append(self.state_translation(state))
        self.next_state_list.append(self.state_translation(next_state))
        self.action_list.append(action)
        self.rewards_list.append(reward)
        self.dones.append(int(done))

    def reset_buffer(self):
        """Clear stored transitions at end of episode."""
        self.state_list.clear()
        self.next_state_list.clear()
        self.action_list.clear()
        self.rewards_list.clear()
        self.dones.clear()
