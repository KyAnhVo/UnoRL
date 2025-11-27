from agents.state_translator import STRAT_STATE_DIM_COUNT, strategic_state_translate, strat_state_reward
from agents.deep_uno_agent import DeepUnoAgent
from agents.deeprl_nn import DeepRL_NN

from typing import Collection, override, List, Tuple
import torch

class DeepQStratAgent(DeepUnoAgent):
    target_nn: DeepRL_NN

    # sync target_nn param using online_nn after
    # every SYNC_RATE episodes
    SYNC_RATE: int
    
    GAIN_CARD_PENALTY: float
    LOSE_CARD_REWARD: float

    def __init__(self):
        super().__init__(state_dim=STRAT_STATE_DIM_COUNT)
        self.target_nn = DeepRL_NN(state_dim=STRAT_STATE_DIM_COUNT,
                                   action_dim=61)
        self.target_nn.load_state_dict(self.online_nn.state_dict())

        # lower == more unstable model
        self.SYNC_RATE = 100

        self.GAIN_CARD_PENALTY = 0.02
        self.LOSE_CARD_REWARD = 0.02

    # ------------------------------------------------------
    # RLCard-required API
    # ------------------------------------------------------

    @override
    def step(self, state)->int:
        curr_state = self.state_translation(state)
        q_values = self.online_nn.forward(
                torch.tensor(curr_state,
                             dtype= torch.float32,
                             device = self.online_nn.device))
        legal: List[int] = state['legal_actions']
        
        mask = torch.full_like(q_values, float('-inf'))
        mask[legal] = 0.0

        masked_q = q_values + mask

        action = torch.argmax(masked_q).item()

        # update values for training lists
        # calculate reward
        # append state into state_list and next_state_list
        # append action into action_list
        reward = strat_state_reward(
                self.state_list[-1],
                curr_state,
                lose_card_reward= self.LOSE_CARD_REWARD,
                gain_card_penalty= self.GAIN_CARD_PENALTY
                )

        self.record_transition(
                state=curr_state,
                action=int(action),
                reward=reward,
                next_state=curr_state,
                done=False
                )

        return int(action)

    @override
    def eval_step(self, state)->Tuple[int, Collection]:
        return self.step(state), []

        

    # ------------------------------------------------------
    # Required for training
    # ------------------------------------------------------
    
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
        if self.episode_count % self.SYNC_RATE == self.SYNC_RATE - 1:
            self.target_nn.load_state_dict(self.online_nn.state_dict())
