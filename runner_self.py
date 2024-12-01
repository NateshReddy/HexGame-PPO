import random
import torch
from ourhexenv import OurHexGame
from fhtw_hex.ppo_smaller import Agent
from tqdm import tqdm
from fhtw_hex.bit_smarter_agent import BitSmartAgent

# Parameters
MODEL_PATH_OLD = "ppo_checkpoint.pth"
MODEL_PATH_NEW = "ppo_checkpoint_agent2.pth"
NUM_GAMES = 100  # Number of games to evaluate

# Initialize environment
env = OurHexGame(board_size=11, render_mode=None, sparse_flag=False)  # No rendering for speed

# Initialize Old PPO Agent
ppo_agent_old = Agent(
    n_actions=env.action_spaces[env.possible_agents[0]].n,
    input_dims=[env.board_size * env.board_size],
    gamma=0.99,
    alpha=0.0003,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_epochs=10
)

# Load trained Old PPO model
try:
    checkpoint = torch.load(MODEL_PATH_OLD)
    ppo_agent_old.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    print(f"Old model loaded successfully from {MODEL_PATH_OLD}.")
except FileNotFoundError:
    print(f"Old model file not found at {MODEL_PATH_OLD}. Proceeding without old PPO agent.")
ppo_agent_old.actor_critic.eval()

# Initialize New PPO Agent
ppo_agent_new = Agent(
    n_actions=env.action_spaces[env.possible_agents[0]].n,
    input_dims=[env.board_size * env.board_size],
    gamma=0.99,
    alpha=0.0003,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_epochs=10
)

# Load trained New PPO model
try:
    checkpoint = torch.load(MODEL_PATH_NEW)
    ppo_agent_new.actor_critic.load_state_dict(checkpoint['model_state_dict'])  # Load the actor model
    print(f"New model loaded successfully from {MODEL_PATH_NEW}.")
except FileNotFoundError:
    print(f"New model file not found at {MODEL_PATH_NEW}. Proceeding with untrained PPO agent.")
ppo_agent_new.actor_critic.eval()


# Define the run_game function
def run_game(agent_1, agent_2, env, agent_1_player=1):
    """
    Runs a single game between two PPO agents.

    Args:
        agent_1: The first PPO agent.
        agent_2: The second PPO agent.
        env: The Hex game environment.
        agent_1_player: The player number (1 or 2) assigned to agent_1.

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
        if current_player == agent_1_player:
            obs_flat = observation["observation"].flatten()
            action, _, _ = agent_1.choose_action(obs_flat, info)  # Agent 1 chooses an action
        else:
            obs_flat = observation["observation"].flatten()
            action, _, _ = agent_2.choose_action(obs_flat, info)  # Agent 2 chooses an action

        env.step(action.item())

    # Determine winner
    return 1 if rewards[1] > rewards[2] else 2


# Define the evaluate_agents function
def evaluate_agents(agent_1, agent_2, env, num_games=50):
    """
    Evaluates the performance of two PPO agents over a specified number of games.

    Args:
        agent_1: The first PPO agent.
        agent_2: The second PPO agent.
        env: The Hex game environment.
        num_games: Number of games to play for evaluation.

    Returns:
        float: The win rate of Agent 1 against Agent 2.
    """
    agent_1_wins = 0
    for _ in tqdm(range(num_games)):
        agent_1_player = random.choice([1, 2])  # Randomly assign Agent 1 as player 1 or 2
        winner = run_game(agent_1, agent_2, env, agent_1_player=agent_1_player)
        if winner == agent_1_player:
            agent_1_wins += 1
    
    win_rate = agent_1_wins / num_games
    return win_rate

# Run the evaluation
win_rate = evaluate_agents(ppo_agent_new, ppo_agent_old, env, num_games=NUM_GAMES)
print(f"New PPO Agent Win Rate against Old PPO Agent: {win_rate * 100:.2f}% over {NUM_GAMES} games.")