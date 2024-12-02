from ourhexenv import OurHexGame
from g06agent import G06Agent
from fhtw_hex.bit_smarter_agent import BitSmartAgent
import random

env = OurHexGame(board_size=11)
env.reset()

# player 1
g06agent1 = G06Agent(env)
# player 2
g06agent2 = BitSmartAgent()


done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break

        
        if agent == 'player_1':
            action = g06agent2.select_action(env, info)
        else:
            action = g06agent1.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()