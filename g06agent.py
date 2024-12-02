from typing import Any
import os
from dotenv import load_dotenv
from ourhexenv import OurHexGame
from ppo_hex.ppo_smaller import Agent

class G06Agent(Agent):
    """Wrapper class for our agents.
    """
    def __init__(self, env: OurHexGame) -> "G06Agent":
        load_dotenv()
        if env.sparse_flag:
            agent_file = os.environ.get("G06_SPARSE", None)
            if not agent_file:
                raise ValueError("G06_SPARSE environment variable not set")
            self.agent = Agent.from_file(agent_file, env=env)
        else:
            agent_file = os.environ.get("G06_DENSE", None)
            if not agent_file:
                raise ValueError("G06_DENSE environment variable not set")
            self.agent = Agent.from_file(agent_file, env=env)
            self.env = env

    def select_action(self, observation, reward, termination, truncation, info):
        return self.agent.select_action(observation, reward, termination, truncation, info)