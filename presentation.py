#!/usr/bin/env python3

from env import play_game
from presentation_agents import DeepQCardAgentPresentatiion, DeepQStratAgentPresentatiion, DeepMCCardAgengPresentation, DeepMCStratAgentPresentation

qstrat = DeepQStratAgentPresentatiion()
qcard = DeepQCardAgentPresentatiion()
mcstrat = DeepMCStratAgentPresentation()
mccard = DeepMCCardAgengPresentation()

input("1st game: qstrat vs qcard")

play_game([qstrat, qcard], is_training=False)

print("\033[2J\033[H")

input("2nd game: mcstrat vs mccard")

play_game([mcstrat, mccard], is_training=False)



