from ourhexenv import OurHexGame  # Import your custom environment
from ppo_hex.ppo_smaller import Agent  # Import the PPO implementation from ppo_smaller
from tqdm import tqdm
import torch
from ppo_hex.random_agent import RandomAgent
from ppo_hex.bit_smarter_agent import BitSmartAgent

def swap_roles(env, agent1, agent2):
    """Swap roles of Player 1 and Player 2."""
    temp = agent1
    agent1 = agent2
    agent2 = temp
    env.possible_agents = ["player_2", "player_1"]
    return agent1, agent2


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


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """
    Load the PPO agent's state from a checkpoint file.

    Args:
        agent: The PPO agent to load.
        filename: Name of the checkpoint file.
    """
    try:
        checkpoint = torch.load(filename)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])  # Load the actor model
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def train_against_agent(env, ppo_agent, opponent_agent, episodes):
    """
    Train a PPO agent (Agent1) against a specified opponent (Agent2).

    Args:
        env: The environment for training.
        ppo_agent: The PPO agent being trained (Player 1).
        opponent_agent: The opponent agent (Player 2).
        episodes: Number of training episodes.

    Returns:
        None
    """
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        env.reset()
        terminations = {agent: False for agent in env.possible_agents}
        scores = {agent: 0 for agent in env.possible_agents}

        # Determine roles based on episode number
        if episode < episodes // 2:
            p1_agent, p2_agent = ppo_agent, opponent_agent  # Agent1 as player_1
        else:
            p1_agent, p2_agent = opponent_agent, ppo_agent  # Agent1 as player_2


        while not all(terminations.values()):
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                # Determine current agent
                current_agent = p1_agent if agent_id == "player_1" else p2_agent
                
                if current_agent == ppo_agent:
                    # PPO Agent chooses an action
                    obs_flat = observation["observation"].flatten()
                    action, probs, value = ppo_agent.choose_action(obs_flat, info)
                    env.step(action.item())

                    # Update experience
                    updated_reward = env.rewards[agent_id]
                    ppo_agent.remember(obs_flat, action, probs, value, updated_reward, done)

                else:
                    # Opponent chooses an action
                    obs_flat = observation["observation"].flatten()
                    if isinstance(opponent_agent, Agent):  # PPO self-play
                        action, _, _ = opponent_agent.choose_action(obs_flat, info)
                        action = action.item()
                    else:  # Random or BitSmart agents
                        action = opponent_agent.select_action(env, info)
                    env.step(action)

                    # Update rewards
                    updated_reward = env.rewards[agent_id]
                scores[agent_id] += updated_reward
            else:
                env.step(None)  # Step terminated agents
            terminations = env.terminations

        # Train PPO Agent after each episode
        ppo_agent.learn()

def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent Initialization
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.9755772764511272, 
        actor_lr=0.0008411496052327293, 
        critic_lr = 7.1838488331643e-05, 
        gae_lambda= 0.9092443406919775, 
        policy_clip=0.27771992898889997,
        batch_size=32,
        n_epochs=15
    )

    # Define opponents as a list of tuples (opponent_agent, filename_suffix, episodes)
    opponents = [
        (RandomAgent(), 'random', 2000),
        (BitSmartAgent(), 'bitsmart', 4000)
    ]

    # Train PPO Agent against each opponent
    for opponent_agent, suffix, episodes in opponents:
        print(f"\nTraining against {suffix.capitalize()}Agent...")
        train_against_agent(env, ppo_agent, opponent_agent, episodes=episodes)
        checkpoint_filename = f'ppo_checkpoint_after_{suffix}.pth'
        save_ppo_checkpoint(ppo_agent, filename=checkpoint_filename)

    # Initialize new PPO Agent for self-play
    print("\nInitializing second PPO agent for self-play...")
    agent2 = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.9755772764511272, 
        actor_lr=0.0008411496052327293, 
        critic_lr = 7.1838488331643e-05, 
        gae_lambda= 0.9092443406919775, 
        policy_clip=0.27771992898889997,
        batch_size=32,
        n_epochs=15
    )
    load_ppo_checkpoint(agent2, filename='ppo_checkpoint_after_bitsmart.pth')

    # Self-play training
    print("\nTraining through self-play...")
    train_against_agent(env, ppo_agent, agent2, episodes=8000)

    # Save the final self-play model
    save_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_final.pth', iteration=1000)
    print("Training completed.")


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
                if agent_id == "player_2":
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
        if scores["player_2"] > scores["player_1"]:
            wins += 1
    win_rate = wins / episodes
    print(f"Evaluation completed: Win rate = {win_rate * 100:.2f}%")
    return win_rate


if __name__ == "__main__":
    main()
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # PPO Agent Initialization
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.9755772764511272, 
        actor_lr=0.0008411496052327293, 
        critic_lr = 7.1838488331643e-05, 
        gae_lambda= 0.9092443406919775, 
        policy_clip=0.27771992898889997,
        batch_size=32,
        n_epochs=15
    )

    load_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_final.pth')
    # Evaluate final model
    evaluate_agent(env, ppo_agent)