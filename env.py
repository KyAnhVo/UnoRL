import rlcard
from rlcard import models
from rlcard.agents.random_agent import RandomAgent
from typing import List
import random

# Please freely tune
TEST_EPOCH = 500

from agents.deep_uno_agent import DeepUnoAgent

def get_rule_based_agent():
    uno_rule_model = models.load('uno-rule-v1')
    return uno_rule_model.agents[0]

def play_game(agents: List, is_training: bool):
    env = rlcard.make('uno', config={
            'record_action': True,
        })
    env.set_agents(agents)
    for index, agent in enumerate(agents):
        if isinstance(agent, DeepUnoAgent):
            agent.before_game()
    _, payoff = env.run(is_training)
    for index, agent in enumerate(agents):
        if isinstance(agent, DeepUnoAgent):
            agent.after_game(payoff=payoff[index])

def play_games(agents: List, is_training: bool):
    # divide agents into random pairs
    shuffled = agents.copy()
    if len(shuffled) % 2 != 0:
        additional_agent = get_rule_based_agent()
        shuffled.append(additional_agent)
    random.shuffle(shuffled)
    pairs = [shuffled[i:i+2] for i in range(0, len(shuffled), 2)]

    # force each pair against each other
    for pair in pairs:
        if isinstance(pair[0], DeepUnoAgent) or isinstance(pair[1], DeepUnoAgent):
            play_game(pair, is_training)

BOT_PHASE_GAMES = 500000
def train(total_games: int, training_agents: List[DeepUnoAgent]):
    rlcard_agents = []
    for _ in range(len(training_agents)):
        rlcard_agents.append(get_rule_based_agent())
        rlcard_agents.append(RandomAgent(61))
    all_agents = training_agents + rlcard_agents

    for game_idx in range(total_games):
        if game_idx < BOT_PHASE_GAMES:
            # phase 1: learn vs bots + each other
            play_games(all_agents, is_training=True)
        else:
            # phase 2: primarily self-play (either full list or separate env)
            play_games(training_agents, is_training=True)
