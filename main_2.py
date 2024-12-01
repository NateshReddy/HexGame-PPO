import optuna
from ourhexenv import OurHexGame
from fhtw_hex.ppo_smaller import Agent
from fhtw_hex.reward_utils import compute_rewards, can_win_next_move
import numpy as np
import torch
from tqdm import tqdm
from fhtw_hex.random_agent import RandomAgent
from fhtw_hex.bit_smarter_agent import BitSmartAgent

def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    checkpoint = {
        'model_state_dict': agent.actor_critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_critic.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.actor_critic.critic_optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def objective(trial):
    # Hyperparameters to tune
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1e-2)
    critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1e-2)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)
    policy_clip = trial.suggest_float('policy_clip', 0.1, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)

    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=batch_size,
        n_epochs=n_epochs
    )

    randomAgent = RandomAgent()
    smart_agent = BitSmartAgent()
    n_games = 500  # Reduced number of games for faster tuning

    total_reward = 0
    for game in range(n_games):
        env.reset()
        terminations = {agent: False for agent in env.possible_agents}
        scores = {agent: 0 for agent in env.possible_agents}

        while not all(terminations.values()):
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                if agent_id == "player_1":
                    obs_flat = observation["observation"].flatten()
                    action, probs, value = agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    agent.remember(obs_flat, action, probs, value, updated_reward, done)
                    scores[agent_id] += updated_reward
                elif agent_id == "player_2":
                    action = smart_agent.select_action(env, info)
                    env.step(action)
                    scores[agent_id] += env.rewards[agent_id]

            terminations = env.terminations

        agent.learn()
        total_reward += scores["player_1"]

    average_reward = total_reward / n_games
    return average_reward

def main():
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    # print("Best trial:")
    # trial = study.best_trial
    # print("Value: ", trial.value)
    # print("Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # # Train with the best hyperparameters
    # best_params = study.best_params
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)
    # best_agent = Agent(
    #     n_actions=env.action_spaces[env.possible_agents[0]].n,
    #     input_dims=[env.board_size * env.board_size],
    #     **best_params
    # )
    best_agent = Agent(
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

    random_agent = RandomAgent()
    smart_agent = BitSmartAgent()

    # Train the best agent (you can adjust the number of games)
    n_games = 2000
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
                    action, probs, value = best_agent.choose_action(obs_flat, info)

                    # Step the environment with the chosen action
                    env.step(action.item())

                    # Get the updated reward after the step
                    updated_reward = env.rewards[agent_id]
                    # if env.terminations[agent_id]:
                    #     player_2_rewards.append(-61)
                    #     scores["player_2"] += -61


                    # Store the experience in the PPO agent
                    best_agent.remember(obs_flat, action, probs, value, updated_reward, done)

                    # Update rewards for debugging
                    player_1_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward

                elif agent_id == "player_2":
                    # Randomly sample action for Player 2
                    # action = env.action_space(agent_id).sample(info["action_mask"])
                    action = smart_agent.select_action(env, info)

                    # Step the environment with the chosen action
                    env.step(action)
                    # if env.terminations[agent_id]:
                    #     player_1_rewards.append(-61)
                    #     scores["player_1"] += -61

                    # Get the updated reward after the step
                    updated_reward= env.rewards[agent_id]

                    # Update rewards for debugging
                    player_2_rewards.append(updated_reward)
                    scores[agent_id] += updated_reward
            # Update terminations
            terminations = env.terminations

        # Train the agent after the episode
        best_agent.learn()

        # Print cumulative rewards for debugging
        # print(f"Episode {game + 1}/{n_games}")
        # print(f"Player 1 Total Rewards: {player_1_rewards}")
        # print(f"Player 2 Total Rewards: {player_2_rewards}")
    # Save the final model after all episodes
    save_ppo_checkpoint(best_agent, filename='ppo_checkpoint.pth', iteration=n_games)

if __name__ == "__main__":
    main()