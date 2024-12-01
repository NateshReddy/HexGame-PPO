import random
import torch
from ourhexenv import OurHexGame
from fhtw_hex.ppo_smaller import Agent
from tqdm import tqdm
from fhtw_hex.bit_smarter_agent import BitSmartAgent
from fhtw_hex.random_agent import RandomAgent

# Parameters
MODEL_PATH = "ppo_checkpoint_final.pth"
NUM_GAMES = 100  # Number of games to evaluate

# Initialize environment
env = OurHexGame(board_size=11, render_mode=None, sparse_flag=False)  # No rendering for speed

# Initialize PPO Agent
ppo_agent = Agent(
    n_actions=env.action_spaces[env.possible_agents[0]].n,
    input_dims=[env.board_size * env.board_size],
    gamma=0.99,
    alpha=0.0003,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_epochs=10
)

# Load trained PPO model
try:
    checkpoint = torch.load(MODEL_PATH)
    ppo_agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])  # Load the actor model
    print(f"Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}. Exiting.")
    exit()

ppo_agent.actor_critic.eval()

# Define the run_game function
def run_game(agent_1, agent_2, env, agent_1_player=1):
    env.reset()
    done = False
    rewards = {1: 0, 2: 0}

    while not done:
        observation, reward, terminated, truncated, info = env.last()
        current_player = 1 if env.agent_selection == "player_1" else 2
        rewards[current_player] += reward
        done = terminated or truncated
        if done:
            break

        obs_flat = observation["observation"].flatten()
        if current_player == agent_1_player:
            action, _, _ = agent_1.choose_action(obs_flat, info)
        else:
            action = agent_2.select_action(env, info) if isinstance(agent_2, (RandomAgent, BitSmartAgent)) else agent_2.choose_action(obs_flat, info)[0]
        env.step(action.item() if isinstance(action, torch.Tensor) else action)

    return 1 if rewards[1] > rewards[2] else 2

# Evaluate agents
def evaluate_agents(agent_1, opponents, env, num_games=50):
    for opponent, name in opponents:
        print(f"\nEvaluating against {name.capitalize()}Agent...")
        agent_1_wins = 0
        for _ in tqdm(range(num_games), desc=f"Games against {name.capitalize()}"):
            agent_1_player = random.choice([1, 2])
            winner = run_game(agent_1, opponent, env, agent_1_player=agent_1_player)
            if winner == agent_1_player:
                agent_1_wins += 1
        win_rate = agent_1_wins / num_games
        print(f"PPO Agent Win Rate against {name.capitalize()}Agent: {win_rate * 100:.2f}% over {num_games} games.")

# List of opponents
opponents = [
    (RandomAgent(), 'random'),
    (BitSmartAgent(), 'bitsmart'),
    (ppo_agent, 'self-play')  # Use the same trained agent for self-play evaluation
]

# Run evaluations
evaluate_agents(ppo_agent, opponents, env, num_games=NUM_GAMES)