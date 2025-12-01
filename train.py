#!/usr/bin/env python3

import sys
import csv
import env
from agents.deep_uno_agent import DeepUnoAgent
from agents.deepq_strat import DeepQStratAgent
from agents.deepq_card import DeepQCardAgent
from agents.deepmc_card import DeepMCCardAgent
from agents.deepmc_strat import DeepMCStratAgent
from typing import List

def note_training_game_results(agent: DeepUnoAgent, filename: str):
    with open(filename, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['bucket', 'wins', 'win_rate'])
        for i, wins in enumerate(agent.win_list):
            wr = f"{wins / agent.ACCUMULATE_WIN_COUNT * 100:.2f}"
            writer.writerow([i, wins, wr])

def test_deepq_strat():
    if len(sys.argv) != 2:
        print("Usage: ./main.py <epoch_count>")

    deepq_card      = DeepQCardAgent()
    deepq_strat     = DeepQStratAgent()
    deepmc_card     = DeepMCCardAgent()
    deepmc_strat    = DeepMCStratAgent()

    training_agents: List[DeepUnoAgent] = []
    training_agents.append(deepq_card)
    training_agents.append(deepq_strat)
    training_agents.append(deepmc_card)
    training_agents.append(deepmc_strat)

    epoch_count = int(sys.argv[1])
    env.train(training_agents=training_agents, total_games=epoch_count)

    for agent in training_agents:
        note_training_game_results(agent=agent, filename=f"statistics/win_{agent.FILE_NAME}")

    print("Finished training, statistics:")
    print(f"games played = {epoch_count}")

if __name__ == "__main__":
    # rlcard_test()
    test_deepq_strat()
