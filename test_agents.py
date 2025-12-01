from typing import override
from agents.deepmc_card import DeepMCCardAgent
from agents.deepmc_strat import DeepMCStratAgent
from agents.deepq_card import DeepQCardAgent
from agents.deepq_strat import DeepQStratAgent
from agents.state_translator import int_to_action
import os
import torch

def find_max_suffix_file(folder_path: str, prefix: str):
    best_file = None
    best_num = -1

    for name in os.listdir(folder_path):
        # Must start with "prefix_"
        if not name.startswith(prefix + "_"):
            continue

        # Extract part after last "_"
        suffix = name.rsplit("_", 1)[-1]

        # Remove extension if present
        suffix = suffix.split(".", 1)[0]

        # Check that suffix is numeric
        if not suffix.isdigit():
            continue

        num = int(suffix)
        if num > best_num:
            best_num = num
            best_file = name
    
    return best_file


class DeepMCCardFrozenAgent(DeepMCCardAgent):
    def __init__(self):
        super().__init__()
        self.test_win_count = 0
        self.test_game_count = 0

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="mccard")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
            # self.target_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

    @override
    def step(self, state):
        return self.eval_step(state)[0]

    @override
    def eval_step(self, state):
        return int_to_action(self._greedy_step(state)), []

    @override
    def before_game(self):
        return

    @override
    def after_game(self, payoff: int):
        self.test_win_count += 1 if payoff == 1 else 0
        self.test_game_count += 1

    def reset_win_count(self):
        self.test_win_count = 0
        self.test_game_count = 0


class DeepMCStratFrozenAgent(DeepMCStratAgent):
    def __init__(self):
        super().__init__()
        self.test_win_count = 0
        self.test_game_count = 0

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="mcstrat")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
            # self.target_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

    @override
    def step(self, state):
        return self.eval_step(state)[0]

    @override
    def eval_step(self, state):
        return int_to_action(self._greedy_step(state)), []

    @override
    def before_game(self):
        return

    @override
    def after_game(self, payoff: int):
        self.test_win_count += 1 if payoff == 1 else 0
        self.test_game_count += 1

    def reset_win_count(self):
        self.test_win_count = 0
        self.test_game_count = 0


class DeepQCardFrozenAgent(DeepQCardAgent):
    def __init__(self):
        super().__init__()
        self.test_win_count = 0
        self.test_game_count = 0

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="qcard")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
            self.target_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

    @override
    def step(self, state):
        return self.eval_step(state)[0]

    @override
    def eval_step(self, state):
        return int_to_action(self._greedy_step(state)), []

    @override
    def before_game(self):
        return

    @override
    def after_game(self, payoff: int):
        self.test_win_count += 1 if payoff == 1 else 0
        self.test_game_count += 1

    def reset_win_count(self):
        self.test_win_count = 0
        self.test_game_count = 0

class DeepQStratFrozenAgent(DeepQStratAgent):
    def __init__(self):
        super().__init__()
        self.test_win_count = 0
        self.test_game_count = 0

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="qstrat")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
            self.target_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

    @override
    def step(self, state):
        return self.eval_step(state)[0]

    def eval_step(self, state):
        return int_to_action(self._greedy_step(state)), []

    @override
    def before_game(self):
        return

    @override
    def after_game(self, payoff: int):
        self.test_win_count += 1 if payoff == 1 else 0
        self.test_game_count += 1

    def reset_win_count(self):
        self.test_win_count = 0
        self.test_game_count = 0
