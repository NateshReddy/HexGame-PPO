import random
import torch
from ourhexenv import OurHexGame
from fhtw_hex.ppo_smaller import Agent
from tqdm import tqdm
from fhtw_hex.bit_smarter_agent import BitSmartAgent
from fhtw_hex.random_agent import RandomAgent

# Parameters
MODEL_PATH = "ppo_checkpoint.pth"
NUM_GAMES = 100  # Number of games to evaluate

# Initialize environment
env = OurHexGame(board_size=11, render_mode=None, sparse_flag=False)  # No rendering for speed

# Initialize PPO Agent
ppo_agent = Agent(
    n_actions=env.action_spaces[env.possible_agents[0]].n,
    input_dims=[env.board_size * env.board_size],
    gamma=0.99,
    actor_lr=0.0003,
    critic_lr=0.0003,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_epochs=10
)

bitSmartAgent = BitSmartAgent()
randomAgent = RandomAgent()

# Load trained PPO model
try:
    checkpoint = torch.load(MODEL_PATH)
    ppo_agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])  # Load the actor model
    print(f"Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}. Running with untrained PPO agent.")
ppo_agent.actor.eval()

# Define the run_game function
def run_game(ppo_agent, env, ppo_player=1):
    """
    Runs a single game between the PPO agent and a Random agent.

    Args:
        ppo_agent: The trained PPO agent.
        env: The Hex game environment.
        ppo_player: The player number (1 or 2) assigned to the PPO agent.

    Returns:
        int: The player number (1 or 2) that won the game.
    """
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

        # Select action based on the player
        if current_player == ppo_player:
            obs_flat = observation["observation"].flatten()
            action, _, _ = ppo_agent.choose_action(obs_flat, info)  # PPO agent chooses an action
            action = action.item()
        else:
            action = bitSmartAgent.select_action(env, info)

        env.step(action)

    # Determine winner
    return 1 if rewards[1] > rewards[2] else 2

# Define the evaluate_agent function
def evaluate_agent(ppo_agent, env, num_games=50):
    """
    Evaluates the PPO agent's performance over a specified number of games.

    Args:
        ppo_agent: The trained PPO agent.
        env: The Hex game environment.
        num_games: Number of games to play for evaluation.

    Returns:
        float: The win rate of the PPO agent.
    """
    ppo_wins = 0
    for _ in tqdm(range(num_games)):
        ppo_player = random.choice([1, 2])  # Randomly assign PPO agent as player 1 or 2
        winner = run_game(ppo_agent, env, ppo_player=ppo_player)
        if winner == ppo_player:
            ppo_wins += 1
    
    win_rate = ppo_wins / num_games
    return win_rate

# Run the evaluation
win_rate = evaluate_agent(ppo_agent, env, num_games=NUM_GAMES)
print(f"PPO Agent Win Rate: {win_rate * 100:.2f}% over {NUM_GAMES} games.")