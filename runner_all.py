from ourhexenv import OurHexGame  # Import your custom environment
from fhtw_hex.ppo_smaller import Agent  # Import the PPO implementation from ppo_smaller
from fhtw_hex.random_agent import RandomAgent
from fhtw_hex.bit_smarter_agent import BitSmartAgent
from tqdm import tqdm
import torch


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """Load the PPO agent's state from a checkpoint file."""
    try:
        checkpoint = torch.load(filename)
        agent.actor.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def evaluate_agent(env, agent1, agent2, num_games):
    """
    Evaluate a PPO agent (Agent1) against a specified opponent (Agent2).

    Args:
        env: The Hex game environment.
        agent1: The PPO agent being tested (Player 1).
        agent2: The opponent agent (Player 2).
        num_games: Number of games to play.

    Returns:
        float: The win rate of Agent1 against Agent2.
    """
    agent1_wins = 0
    for _ in tqdm(range(num_games), desc="Evaluating"):
        env.reset()
        terminations = {agent: False for agent in env.possible_agents}
        scores = {agent: 0 for agent in env.possible_agents}

        while not all(terminations.values()):
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                obs_flat = observation["observation"].flatten()
                if agent_id == "player_1":
                    action, _, _ = agent1.choose_action(obs_flat, False)
                    env.step(action)
                elif agent_id == "player_2":
                    if isinstance(agent2, Agent):
                        action, _, _ = agent2.choose_action(obs_flat, False)
                        action = action
                    else:
                        action = agent2.select_action(env, info)
                    env.step(action)
            else:
                env.step(None)  # Step terminated agents

            terminations = env.terminations

        # Determine the winner
        scores = env.rewards
        if scores["player_1"] > scores["player_2"]:
            agent1_wins += 1

    win_rate = agent1_wins / num_games
    return win_rate


def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent Initialization for testing
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n - 1,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )
    load_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_final.pth')

    # Define opponents
    opponents = [
        (RandomAgent(), 'RandomAgent'),
        (BitSmartAgent(), 'BitSmartAgent'),
        (ppo_agent, 'Self-Play Agent')  # Self-play using the trained PPO agent
    ]

    # Evaluate PPO Agent against all opponents
    num_games = 100
    for opponent, name in opponents:
        print(f"\nEvaluating against {name}...")
        win_rate = evaluate_agent(env, ppo_agent, opponent, num_games)
        print(f"Win Rate against {name}: {win_rate * 100:.2f}%")


if __name__ == "__main__":
    main()