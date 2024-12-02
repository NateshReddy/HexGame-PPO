from typing import Any
import os
from ourhexenv import OurHexGame
from ppo_hex.ppo_agent import Agent

class G06Agent(Agent):
    """Wrapper class for our agents.
    """
    def __init__(self, env: OurHexGame) -> "G06Agent":
        if env.sparse_flag:
            agent_file = '/Users/natesh/SJSU/SEM 3/RL/PA5/hexgame-ppo/ppo_checkpoint_final.pth' #os.environ.get("G06_SPARSE", None)
            self.agent = Agent.from_file(agent_file, env=env)
        else:
            agent_file = '/Users/natesh/SJSU/SEM 3/RL/PA5/hexgame-ppo/ppo_checkpoint_final.pth' #os.environ.get("G06_DENSE", None)
            self.agent = Agent.from_file(agent_file, env=env)
            self.env = env

    def select_action(self, observation, reward, termination, truncation, info):
        return self.agent.select_action(observation, reward, termination, truncation, info)