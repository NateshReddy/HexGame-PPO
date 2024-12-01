from ourhexenv import OurHexGame  # Import your custom environment
from fhtw_hex.ppo_smaller import Agent  # Import the PPO implementation from ppo_smaller
from fhtw_hex.reward_utils import compute_rewards, can_win_next_move  # Updated rewards utility
import numpy as np
import torch
from tqdm import tqdm

def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    """
    Save the PPO agent's state in a checkpoint file.

    Args:
        agent: The PPO agent to save.
        filename: Name of the checkpoint file.
        iteration: Current training iteration.
    """
    checkpoint = {
        'model_state_dict': agent.actor_critic.state_dict(),  # Save actor model state
        'actor_optimizer_state_dict': agent.actor.optimizer.state_dict(),  # Save actor optimizer state
        'critic_optimizer_state_dict': agent.critic.optimizer.state_dict(),  # Save critic optimizer state
        'iteration': iteration  # Store current iteration or epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """
    Load the PPO agent's state from a checkpoint file.

    Args:
        agent: The PPO agent to load.
        filename: Name of the checkpoint file.
    """
    try:
        checkpoint = torch.load(filename)
        agent.actor.load_state_dict(checkpoint['model_state_dict'])
        agent.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent 1: Load from a checkpoint
    agent1 = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n - 1,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )
    load_ppo_checkpoint(agent1, filename='ppo_checkpoint.pth')

    # PPO Agent 2: Train from scratch
    agent2 = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n - 1,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )

    n_games = 50  # Total games to play

    for game in tqdm(range(n_games)):
        env.reset()
        terminations = {agent: False for agent in env.possible_agents}
        scores = {agent: 0 for agent in env.possible_agents}

        # Reward trackers for debugging
        player_1_rewards = []
        player_2_rewards = []

        while not all(terminations.values()):
            # Fetch the current agent
            agent_id = env.agent_selection

            # Fetch the last observation, reward, and termination status
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                if agent_id == "player_1":
                    # Agent 1 chooses an action
                    obs_flat = observation["observation"].flatten()
                    action, probs, value = agent1.choose_action(obs_flat)

                    # Step the environment with the chosen action
                    env.step(action)

                    # Get the updated reward after the step
                    updated_reward = env.rewards[agent_id]

                    # Update rewards for debugging
                    player_1_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward

                elif agent_id == "player_2":
                    # Agent 2 chooses an action
                    obs_flat = observation["observation"].flatten()
                    action, probs, value = agent2.choose_action(obs_flat)

                    # Step the environment with the chosen action
                    env.step(action)

                    # Get the updated reward after the step
                    updated_reward = env.rewards[agent_id]

                    # Store the experience in the PPO agent
                    agent2.remember(obs_flat, action, probs, value, updated_reward, done)

                    # Update rewards for debugging
                    player_2_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward
            else:
                # For terminated agents, step with None
                env.step(None)

            # Update terminations
            terminations = env.terminations

        # Train Agent 2 after the episode
        agent2.learn()

    # Save the final model of Agent 2 after all episodes
    save_ppo_checkpoint(agent2, filename='ppo_checkpoint_agent2.pth', iteration=n_games)
    print("Training completed.")

if __name__ == "__main__":
    main()