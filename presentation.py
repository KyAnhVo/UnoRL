#!/usr/bin/env python3

from env import play_game
from presentation_agents import DeepQCardAgentPresentatiion, DeepQStratAgentPresentatiion

qstrat = DeepQStratAgentPresentatiion()
qcard = DeepQCardAgentPresentatiion()

input("1st game: qstrat vs qcard")

play_game([qstrat, qcard], is_training=False)

