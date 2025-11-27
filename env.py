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
    trajectories, payoff = env.run(is_training)
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

def train(epoch: int, training_agents: List):
    # setup
    rlcard_agents = []
    for i in range(len(training_agents)):
        bot = get_rule_based_agent()
        rlcard_agents.append(bot)
        bot = RandomAgent(61)
        rlcard_agents.append(bot)
    all_agents = training_agents + rlcard_agents
    
    for i in range(epoch):
        is_training = not (i % TEST_EPOCH == 0)
        play_games(all_agents, is_training)
