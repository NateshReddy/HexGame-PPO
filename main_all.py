from ourhexenv import OurHexGame  # Import your custom environment
from fhtw_hex.ppo_smaller import Agent  # Import the PPO implementation from ppo_smaller
from tqdm import tqdm
import torch
from fhtw_hex.random_agent import RandomAgent
from fhtw_hex.bit_smarter_agent import BitSmartAgent


def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    """Save the PPO agent's state in a checkpoint file."""
    checkpoint = {
        'model_state_dict': agent.actor.state_dict(),
        'actor_optimizer_state_dict': agent.actor.optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic.optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """Load the PPO agent's state from a checkpoint file."""
    try:
        checkpoint = torch.load(filename)
        agent.actor.load_state_dict(checkpoint['model_state_dict'])
        agent.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def train_against_agent(env, agent1, agent2, episodes):
    """Train a PPO agent (Agent1) against a specified opponent (Agent2)."""
    for episode in tqdm(range(episodes), desc=f"Training"):
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
                    action, probs, value = agent1.choose_action(obs_flat, False)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    agent1.remember(obs_flat, action, probs, value, updated_reward, done)
                    scores[agent_id] += updated_reward
                elif agent_id == "player_2":
                    if isinstance(agent2, Agent):
                        action, _, _ = agent2.choose_action(obs_flat, False)
                        action = action
                    else:
                        action = agent2.select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)  # Step terminated agents
            terminations = env.terminations

        agent1.learn()


def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent Initialization
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

    # Define opponents
    opponents = [
        (RandomAgent(), 'random', 500),
        (BitSmartAgent(), 'bitsmart', 500)
    ]

    # Train PPO Agent against each opponent
    for opponent, suffix, episodes in opponents:
        print(f"\nTraining against {suffix.capitalize()}Agent...")
        train_against_agent(env, ppo_agent, opponent, episodes)
        save_ppo_checkpoint(ppo_agent, filename=f'ppo_checkpoint_after_{suffix}.pth')

    # Initialize PPO Agent 2 for self-play
    print("\nInitializing PPO agent for self-play...")
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
    load_ppo_checkpoint(agent2, filename='ppo_checkpoint_after_bitsmart.pth')

    # Self-play training
    print("\nTraining through self-play...")
    train_against_agent(env, ppo_agent, agent2, episodes=10000)

    # Save the final self-play model
    save_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_final.pth')
    print("Training completed.")


if __name__ == "__main__":
    main()