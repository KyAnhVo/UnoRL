#!/usr/bin/env python3

from env import play_game, get_rule_based_agent
from rulebot_agents import DeepQStratRulebot, DeepQCardRulebot, DeepMCCardRulebot, DeepMCStratRulebot
from agents.deep_uno_agent import DeepUnoAgent
import sys
import csv
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

    deepq_card      = DeepQCardRulebot()
    deepq_strat     = DeepQStratRulebot()
    deepmc_card     = DeepMCCardRulebot()
    deepmc_strat    = DeepMCStratRulebot()
    rulebot         = get_rule_based_agent()

    training_agents: List[DeepUnoAgent] = []
    training_agents.append(deepq_card)
    training_agents.append(deepq_strat)
    training_agents.append(deepmc_card)
    training_agents.append(deepmc_strat)

    epoch_count = int(sys.argv[1])
    for _ in range(epoch_count):
        for agent in training_agents:
            play_game([agent, rulebot], is_training=True)

    for agent in training_agents:
        note_training_game_results(agent=agent, filename=f"statistics_rulebot/win_{agent.FILE_NAME}")

    print("Finished training, statistics:")
    print(f"games played = {epoch_count}")

if __name__ == "__main__":
    # rlcard_test()
    test_deepq_strat()
