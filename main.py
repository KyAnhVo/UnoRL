import rlcard
from rlcard import models
from rlcard.agents.uno_human_agent import HumanAgent

# If your version exports a shortcut, you *might* instead be able to do:
# from rlcard.agents import UnoHumanAgent as HumanAgent

def rlcard_test():
    # 1. Make UNO environment
    env = rlcard.make('uno', config={
        'seed': 42,
        'record_action': True,
    })

    # 2. Human agent (CLI interface)
    human_agent = MyRandomAgent(env.action_num)

    # 3. Load rule-based UNO model from the model zoo
    #    This returns a Model; `.agents` is a list of agents for each seat.
    uno_rule_model = models.load('uno-rule-v1')
    rule_agents = uno_rule_model.agents   # length = env.num_players (typically 4)

    # Put human at seat 0, keep rule agents for others
    agents = [human_agent] + rule_agents[1:]
    env.set_agents(agents)

    print(">> UNO: you (Player 0) vs rule-based bots")

    while True:
        print("\n>> Start a new game")
        trajectories, payoffs = env.run(is_training=False)

        # payoffs[i] is the final reward for player i
        your_payoff = payoffs[0]
        print("===============   Result   ===============")
        if your_payoff > 0:
            print(f"You win! payoff = {your_payoff}")
        elif your_payoff < 0:
            print(f"You lose. payoff = {your_payoff}")
        else:
            print("Tie game.")

        # Ask if we continue
        cont = input("Play another game? (y/n): ").strip().lower()
        if cont != 'y':
            break
import numpy as np


class MyRandomAgent:
    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        print('================================')
        print(state)
        return np.random.choice(list(state['legal_actions']))

    def eval_step(self, state):
        info = {}
        return self.step(state), info


if __name__ == "__main__":
    rlcard_test()

