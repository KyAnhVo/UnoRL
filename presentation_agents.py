from typing import override, Collection, Tuple

import torch
from agents.deepq_card import DeepQCardAgent
from agents.deepq_strat import DeepQStratAgent
from agents.deepmc_card import DeepMCCardAgent
from agents.deepmc_strat import DeepMCStratAgent


import shutil
import os

TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(80, 24))

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

class DeepQCardAgentPresentatiion(DeepQCardAgent):
    def __init__(self):
        super().__init__()
        self.presentation_line = "DQN: Card Presentation"

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

        self.win = 0

    @override
    def eval_step(self, state) -> Tuple[str, Collection]:
        # print("\033[2J\033[H")
        print("-"*TERM_WIDTH)
        real_state = state['raw_obs']

        print(self.presentation_line)
        print(f"Hand count: {len(real_state['hand'])}")
        print(f"Hand: {real_state['hand']}")
        print(f"Target: {real_state['target']}")
        print(f"Opponent hand count: {len(real_state['others_hand'])}")
        print(f"Discarded deck count: {len(real_state['played_cards'])}")
        print()

        action, _ = super().eval_step(state)
        
        print()
        print(f"Play: {action}")
        print()

        return action, []

    @override
    def after_game(self, payoff: int):
        # print("\033[2J\033[H")
        print('-' * TERM_WIDTH)
        print("DQN Card WIN!" if payoff == 1 else "DQN Card LOSE T.T")
        self.win += 1 if payoff == 1 else 0
        # input('')
        # print("\033[2J\033[H")

class DeepQStratAgentPresentatiion(DeepQStratAgent):
    def __init__(self):
        super().__init__()
        self.presentation_line = "DQN: Strat Presentation "

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

        self.win = 0

    @override
    def eval_step(self, state) -> Tuple[str, Collection]:
        # print("\033[2J\033[H")
        print("-" * TERM_WIDTH)

        real_state = state['raw_obs']

        print(self.presentation_line)
        print(f"Hand count: {len(real_state['hand'])}")
        print(f"Hand: {real_state['hand']}")
        print(f"Target: {real_state['target']}")
        print(f"Opponent hand count: {len(real_state['others_hand'])}")
        print(f"Discarded deck count: {len(real_state['played_cards'])}")

        print()

        action, _ = super().eval_step(state)
        
        print()
        print(f"Play: {action}")
        print()

        return action, []

    @override
    def after_game(self, payoff: int):
        print("-" * TERM_WIDTH)
        # print("\033[2J\033[H")
        print("DQN Strat WIN!" if payoff == 1 else "DQN Strat LOSE T.T")
        # input('')
        # print("\033[2J\033[H")
        self.win += 1 if payoff == 1 else 0

class DeepMCStratAgentPresentation(DeepMCStratAgent):
    def __init__(self):
        super().__init__()
        self.presentation_line = "DeepMC: Strat Presentation "

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="mcstrat")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

        self.win = 0

    @override
    def eval_step(self, state) -> Tuple[str, Collection]:
        # print("\033[2J\033[H")
        print("-" * TERM_WIDTH)
        real_state = state['raw_obs']

        print(self.presentation_line)
        print(f"Hand count: {len(real_state['hand'])}")
        print(f"Hand: {real_state['hand']}")
        print(f"Target: {real_state['target']}")
        print(f"Opponent hand count: {len(real_state['others_hand'])}")
        print(f"Discarded deck count: {len(real_state['played_cards'])}")

        print()

        action, _ = super().eval_step(state)
        
        print()
        print(f"Play: {action}")
        print()

        return action, []

    @override
    def after_game(self, payoff: int):
        print("-" * TERM_WIDTH)
        # print("\033[2J\033[H")
        print("DeepMC Strat WIN!" if payoff == 1 else "DeepMC Strat LOSE T.T")
        # input('')
        # print("\033[2J\033[H")
        self.win += 1 if payoff == 1 else 0

class DeepMCCardAgengPresentation(DeepMCCardAgent):
    def __init__(self):
        super().__init__()
        self.presentation_line = "DeepMC: Card Presentation "

        file = find_max_suffix_file(
                folder_path=os.path.join(".", "model_history"),
                prefix="mccard")

        if file is not None:
            file = os.path.join(".", "model_history", file)
            self.online_nn.load_state_dict(torch.load(file))
        else:
            print("not found file")
            exit()

        self.win = 0

    @override
    def eval_step(self, state) -> Tuple[str, Collection]:
        #print("\033[2J\033[H")
        print('-' * TERM_WIDTH)
        real_state = state['raw_obs']

        print(self.presentation_line)
        print(f"Hand count: {len(real_state['hand'])}")
        print(f"Hand: {real_state['hand']}")
        print(f"Target: {real_state['target']}")
        print(f"Opponent hand count: {len(real_state['others_hand'])}")
        print(f"Discarded deck count: {len(real_state['played_cards'])}")

        print()

        action, _ = super().eval_step(state)
        
        print()
        print(f"Play: {action}")
        print()

        return action, []

    @override
    def after_game(self, payoff: int):
        print("-" * TERM_WIDTH)
        # print("\033[2J\033[H")
        print("DeepMC Card WIN!" if payoff == 1 else "DeepMC Card LOSE T.T")
        # input('')
        # print("\033[2J\033[H")
        self.win += 1 if payoff == 1 else 0
