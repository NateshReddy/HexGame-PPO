from ourhexenv import OurHexGame
from agent_group3.g03agent import G03Agent
from agent_group12.g12agent import G12Agent
from g06agent import G06Agent
import random

env = OurHexGame(board_size=11, sparse_flag= False)
env.reset()

# player 1
gXXagent = G06Agent(env)
# player 2
gYYagent = G03Agent(env)

smart_agent_player_id = random.choice(env.agents)

rewards = {'player_1': 0, 'player_2': 0}
done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        # rewards[agent] += reward
        if termination or truncation:
            done = True
            break

        if agent == 'player_1':
            action = gXXagent.select_action(observation, reward, termination, truncation, info)
        else:
            action = gYYagent.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()
    
    # print(rewards)
