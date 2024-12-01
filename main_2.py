from ourhexenv import OurHexGame  # Import your custom environment
from fhtw_hex.ppo_smaller import Agent  # Import the PPO implementation from ppo_smaller
from fhtw_hex.reward_utils import compute_rewards, can_win_next_move  # Updated rewards utility
import numpy as np
import ipdb
import torch
from tqdm import tqdm
from fhtw_hex.random_agent import RandomAgent
from fhtw_hex.bit_smarter_agent import BitSmartAgent

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
        'actor_optimizer_state_dict': agent.actor_critic.actor_optimizer.state_dict(),  # Save actor optimizer state
        'critic_optimizer_state_dict': agent.actor_critic.critic_optimizer.state_dict(),  # Save critic optimizer state
        'iteration': iteration  # Store current iteration or epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent Initialization
    agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )

    randomAgent = RandomAgent()
    bitSmartAgent = BitSmartAgent()

    n_games = 5000 # Total games to play

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
            # import ipdb; ipdb.set_trace()
            if not done:
                if agent_id == "player_1":
                    # Choose an action for Player 1
                    obs_flat = observation["observation"].flatten()
                    action, probs, value = agent.choose_action(obs_flat, info)

                    # Step the environment with the chosen action
                    env.step(action.item())

                    # Get the updated reward after the step
                    updated_reward = env.rewards[agent_id]

                    # Store the experience in the PPO agent
                    agent.remember(obs_flat, action, probs, value, updated_reward, done)

                    # Update rewards for debugging
                    player_1_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward

                elif agent_id == "player_2":
                    # Randomly sample action for Player 2
                    # action = env.action_space(agent_id).sample(info["action_mask"])
                    action = bitSmartAgent.select_action(env, info)

                    # Step the environment with the chosen action
                    env.step(action)

                    # Get the updated reward after the step
                    updated_reward= env.rewards[agent_id]

                    # Update rewards for debugging
                    player_2_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward
            else:
                # For terminated agents, step with None
                env.step(None)

            # Update terminations
            terminations = env.terminations

        # Train the agent after the episode
        agent.learn()

        # Print cumulative rewards for debugging
        print(f"Episode {game + 1}/{n_games}")
        print(f"Player 1 Total Rewards: {player_1_rewards}")
        print(f"Player 2 Total Rewards: {player_2_rewards}")
    # Save the final model after all episodes
    save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=n_games)
    # print(f"Training completed. Best score: {max(all_scores)}")

if __name__ == "__main__":
    main()