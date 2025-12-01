#!/usr/bin/env python3

from test_agents import DeepMCCardFrozenAgent, DeepMCStratFrozenAgent, DeepQCardFrozenAgent, DeepQStratFrozenAgent
from env import play_game, get_rule_based_agent
from rlcard.agents.random_agent import RandomAgent

qcard = DeepQCardFrozenAgent()
qstrat = DeepQStratFrozenAgent()
mccard = DeepMCCardFrozenAgent()
mcstrat = DeepMCStratFrozenAgent()

rulebot = get_rule_based_agent()
randbot = RandomAgent(61)

gamecount = 10000
print(f"Games played per round: {gamecount}")

for i in range(gamecount):
    play_game([qcard, rulebot], False)
print(f"qcard vs rulebot: {qcard.test_win_count / qcard.test_game_count * 100:.2f}%")
qcard.reset_win_count()

for i in range(gamecount):
    play_game([qcard, randbot], False)
print(f"qcard vs randbot: {qcard.test_win_count / qcard.test_game_count * 100:.2f}%")
qcard.reset_win_count()

for i in range(gamecount):
    play_game([qstrat, rulebot], False)
print(f"qstrat vs rulebot: {qstrat.test_win_count / qstrat.test_game_count * 100:.2f}%")
qstrat.reset_win_count()

for i in range(gamecount):
    play_game([qstrat, randbot], False)
print(f"qstrat vs randbot: {qstrat.test_win_count / qstrat.test_game_count * 100:.2f}%")
qstrat.reset_win_count()

for i in range(gamecount):
    play_game([mccard, rulebot], False)
print(f"mccard vs rulebot: {mccard.test_win_count / mccard.test_game_count * 100:.2f}%")
mccard.reset_win_count()

for i in range(gamecount):
    play_game([mccard, randbot], False)
print(f"mccard vs randbot: {mccard.test_win_count / mccard.test_game_count * 100:.2f}%")
mccard.reset_win_count()

for i in range(gamecount):
    play_game([mcstrat, rulebot], False)
print(f"mcstrat vs rulebot: {mcstrat.test_win_count / mcstrat.test_game_count * 100:.2f}%")
mcstrat.reset_win_count()

for i in range(gamecount):
    play_game([mcstrat, randbot], False)
print(f"mcstrat vs randbot: {mcstrat.test_win_count / mcstrat.test_game_count * 100:.2f}%")
mcstrat.reset_win_count()
