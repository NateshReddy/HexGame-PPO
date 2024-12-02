import torch
from ourhexenv import OurHexGame
from agents.ppo_agent import Agent
from tqdm import tqdm
from agents.random_agent import RandomAgent
from agents.bit_smarter_agent import BitSmartAgent
# from agent_group3.g03agent import G03Agent  # Import the new agent

def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    """Save the PPO agent's state in a checkpoint file."""
    checkpoint = {
        'model_state_dict': agent.actor_critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_critic.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.actor_critic.critic_optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """Load the PPO agent's state from a checkpoint file."""
    try:
        checkpoint = torch.load(filename)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def train_against_agent(env, agent1, agent2, episodes):
    """
    Train a PPO agent (Agent1) against a specified opponent (Agent2) with role-swapping.

    Args:
        env: The environment for training.
        agent1: The PPO agent being trained.
        agent2: The opponent agent.
        episodes: Total number of training episodes.

    Returns:
        None
    """
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        env.reset()
        done = False
        scores = {agent: 0 for agent in env.possible_agents}

        # Determine roles based on episode number
        if episode < episodes // 2:
            p1_agent, p2_agent = agent1, agent2  # Agent1 as player_1
        else:
            p1_agent, p2_agent = agent2, agent1  # Agent1 as player_2

        while not done:
            agent_id = env.agent_selection
            observation, reward, terminated, truncated, info = env.last()
            done = terminated or truncated

            if not done:
                obs_flat = observation["observation"].flatten()

                # Determine current agent
                current_agent = p1_agent if agent_id == "player_1" else p2_agent

                if isinstance(current_agent, Agent):  # PPO/self-play
                    action, probs, value = current_agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    if current_agent == agent1:
                        agent1.remember(obs_flat, action, probs, value, updated_reward, done)
                    scores[agent_id] += updated_reward
                else:  # RandomAgent or BitSmartAgent
                    action = current_agent.select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)

        # Train PPO Agent after each episode
        agent1.learn()


def evaluate_agent(env, agent, episodes=100):
    """Evaluate the PPO agent against BitSmartAgent."""
    wins = 0
    for _ in tqdm(range(episodes), desc="Evaluation Episodes"):
        env.reset()
        done = False
        scores = {agent: 0 for agent in env.possible_agents}
        while not done:
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            if not done:
                if agent_id == "player_1":
                    obs_flat = observation["observation"].flatten()
                    action, _, _ = agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
                else:
                    action = BitSmartAgent().select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)
        if scores["player_1"] > scores["player_2"]:
            wins += 1
    win_rate = wins / episodes
    print(f"Evaluation completed: Win rate = {win_rate * 100:.2f}%")
    return win_rate


def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=True, render_mode=None)

    # Initialize PPO Agent with fixed parameters
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        actor_lr=0.00022,
        critic_lr=0.0011,
        gae_lambda=0.92,
        policy_clip=0.2,
        batch_size=128,
        n_epochs=18
    )

    # Define opponents
    opponents = [
        (RandomAgent(), 'random', 2000),
        (BitSmartAgent(), 'bitsmart', 2000)
    ]

    # Train PPO Agent against each opponent
    for opponent_agent, suffix, episodes in opponents:
        print(f"\nTraining against {suffix.capitalize()}Agent...")
        train_against_agent(env, ppo_agent, opponent_agent, episodes)
        save_ppo_checkpoint(ppo_agent, filename=f'ppo_checkpoint_after_{suffix}.pth')

    # Initialize PPO Agent 2 for self-play
    print("\nInitializing PPO agent for self-play...")
    agent2 = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        actor_lr=0.00022,
        critic_lr=0.0011,
        gae_lambda=0.92,
        policy_clip=0.2,
        batch_size=128,
        n_epochs=18
    )
    load_ppo_checkpoint(agent2, filename='ppo_checkpoint_after_bitsmart.pth')

    # Self-play training
    print("\nTraining through self-play...")
    train_against_agent(env, ppo_agent, agent2, episodes=5000)

    # Save the intermediate model after self-play
    save_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_after_selfplay.pth')

    # Save the final model
    save_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_sparse.pth', iteration=2000)

    # Evaluate the final model
    # evaluate_agent(env, ppo_agent)


if __name__ == "__main__":
    main()