from types import FunctionType
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepRL_NN(nn.Module):
    state_dim: int
    action_dim: int

    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    activation: FunctionType
    optimizer: torch.optim.Optimizer
    lr: float

    EPOCH_PER_TRAIN: int

    def __init__(self, state_dim: int, action_dim, lr=1e-3):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128, dtype=torch.float32)
        self.fc2 = nn.Linear(128, 64, dtype=torch.float32)
        self.fc3 = nn.Linear(64, action_dim, dtype=torch.float32)

        self.activation = F.leaky_relu
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.EPOCH_PER_TRAIN = 1

    def train_mc_batch(
            self, state_list: List[List[int]],
            actions_taken: List[int],
            returns: List[float]
            )->float:
        ''' Apply batch nn training based on MC learning item queues.

        Args:
            state_list: List of each state s_i
            actions_taken: List of each action taken a_i
            returns: the return value of each (s_i, a_i) pair in OSL
        Returns:
            average loss per epoch in training
        '''
        return self._train_batch(state_list, actions_taken, returns)

    def train_q_batch(
            self, state_list: List[List[int]],
            actions_taken: List[int],
            q_values: List[float]
            )->float:
        ''' Apply batch nn training based on Q learning item queues

        Args:
            state_list: List of each state s_i
            actions_taken: List of each action taken a_i
            q_values: the return value of each pair (s_i, a_i) pair in q learning
        Returns:
            average loss per epoch in training
        '''
        return self._train_batch(state_list, actions_taken, q_values)

    # ================ "PRIVATE" FUNCTIONS ================

    def _train_batch(
            self, state_list: List[List[int]],
            actions_taken: List[int],
            real_values: List[float])->float:
        ''' Apply batch nn training based on any RL learning item queues.

        Args:
            state_list: List of each state s_i
            actions_taken: List of each action taken a_i
            real_labels: list of target values
        Returns:
            average loss per epoch in training
        '''
        states = torch.tensor(
                state_list, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
                actions_taken, dtype=torch.long, device=self.device)
        targets = torch.tensor(
                real_values, dtype=torch.float32, device=self.device)
        total_loss = 0.0

        for _ in range(self.EPOCH_PER_TRAIN):
            predicted_q_values = self.forward(states)
            current_q_values = predicted_q_values.gather(
                    1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
            self.optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / self.EPOCH_PER_TRAIN
        return avg_loss
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

