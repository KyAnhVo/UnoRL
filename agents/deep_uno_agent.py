from abc import ABC, abstractmethod
from typing import List
from agents.deeprl_nn import DeepRL_NN
from random import randint

class DeepUnoAgent(ABC):
    state_dim: int
    gamma: float

    online_nn: DeepRL_NN
    episode_count: int

    # train after every TRAIN_RATE games
    TRAIN_RATE: int

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

        # Smaller == faster training, higher fluctuation
        self.TRAIN_RATE = 10

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
    # Required for training
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

    @abstractmethod
    def before_game(self):
        """ Before-game setup: alter buffer, etc. """

        # adjust buffer
        buffer_state = [0 for _ in range(self.state_dim)]
        self.state_list.append(buffer_state)
        self.action_list.append(randint(0, 60)) # doesnt matter


    @abstractmethod
    def after_game(self, payoff: int):
        """ After-game setup: adjust buffer, training, etc. """
        # adjust buffer
        self.next_state_list.append([0 for _ in range(self.state_dim)])
        self.rewards_list.append(payoff)
        self.dones.append(True)

        # training
        self.episode_count += 1
        if self.episode_count % self.TRAIN_RATE == self.TRAIN_RATE - 1:
            self.train_online_nn()
            self.reset_buffer()


    # ------------------------------------------------------
    # Helpers that SHOULD be included
    # ------------------------------------------------------

    def record_transition(self, 
                          state: List[int], 
                          action: int, 
                          reward: float, 
                          next_state: List[int], 
                          done: bool):
        """Add a transition to buffers."""
        self.state_list.append(self.state_translation(state))
        if next_state != None:
            self.next_state_list.append(self.state_translation(next_state))
        if action != -1:
            self.action_list.append(action)
        self.rewards_list.append(reward)
        self.dones.append(done)

    def reset_buffer(self):
        """Clear stored transitions at end of episode."""
        self.state_list.clear()
        self.next_state_list.clear()
        self.action_list.clear()
        self.rewards_list.clear()
        self.dones.clear()
