from env import play_game, get_rule_based_agent
from presentation_agents import DeepQCardAgentPresentatiion, DeepQStratAgentPresentatiion, TERM_WIDTH

qstrat = DeepQStratAgentPresentatiion()
qcard = DeepQCardAgentPresentatiion()

input("Press enter to start game")

play_game([qstrat, qcard], is_training=False)

