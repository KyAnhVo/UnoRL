import math
from agents.deep_uno_agent import DeepUnoAgent
from agents.deeprl_nn import DeepRL_NN
from typing import Collection, override, List, Tuple
import torch
import random
from agents.state_translator import int_to_action, card_to_int

class DeepQAgent(DeepUnoAgent):
    target_nn: DeepRL_NN

    # sync target_nn param using online_nn after
    # every SYNC_RATE episodes
    SYNC_RATE: int
    
    GAIN_CARD_PENALTY: float
    LOSE_CARD_REWARD: float

    epsilon: float
    EPSILON_MIN: float
    EPSILON_MAX: float
    EPSILON_DECAY_CONSTANT: float

    def __init__(self, state_dim: int):
        super().__init__(state_dim=state_dim)
        self.target_nn = DeepRL_NN(state_dim=state_dim, action_dim=61)
        self.target_nn.load_state_dict(self.online_nn.state_dict())

        # lower == more unstable model
        self.SYNC_RATE = 1000

        self.GAIN_CARD_PENALTY = 0.02
        self.LOSE_CARD_REWARD = 0.02
        
        self.EPSILON_MIN = 0.05
        self.EPSILON_MAX = 1.00
        self.EPSILON_DECAY_CONSTANT = 5e-5

        self.epsilon = self.EPSILON_MAX

    # ------------------------------------------------------
    # RLCard-required API
    # ------------------------------------------------------

    @override
    def step(self, state) -> str:
        curr_state = self.state_translation(state)
        
        # Calculate reward based on previous state
        reward = 0
        if len(self.state_list) > 0:
            reward = self.calculate_reward(self.state_list[-1], curr_state)
        
        # Record transition
        self.record_transition(
            state=curr_state,
            action=0,  # Will be updated below
            reward=reward,
            next_state=curr_state,
            done=False
        )
        
        # Action selection (epsilon-greedy)
        if random.random() < self.epsilon:
            action_str = random.choice(state['raw_legal_actions'])
            action_int = card_to_int(action_str) if action_str != 'draw' else 60
        else:
            q_values = self.online_nn.forward(
                torch.tensor(curr_state, dtype=torch.float32, device=self.online_nn.device)
            )
            legal: List[int] = state['legal_actions']
            
            mask = torch.full_like(q_values, float('-inf'))
            mask[legal] = 0.0
            masked_q = q_values + mask
            
            action_int = torch.argmax(masked_q).item()
            action_str = int_to_action(int(action_int))
        
        # Update the action in the buffer
        self.action_list[-1] = int(action_int)
        
        return action_str

    @override
    def eval_step(self, state) -> Tuple[str, Collection]:
        return self.step(state), []

    # ------------------------------------------------------
    # Required for training
    # ------------------------------------------------------
    
    @override
    def state_translation(self, state) -> List[int]:
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement state_translation")
    
    def calculate_reward(self, prev_state: List[int], curr_state: List[int]) -> float:
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate_reward")

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

    @override
    def before_game(self):
        super().before_game()

    @override
    def after_game(self, payoff: int):
        super().after_game(payoff)
        
        # Epsilon decay
        self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(
            -self.EPSILON_DECAY_CONSTANT * self.episode_count
        )
        
        # Sync target network periodically
        if self.episode_count % self.SYNC_RATE == self.SYNC_RATE - 1:
            self.target_nn.load_state_dict(self.online_nn.state_dict())
