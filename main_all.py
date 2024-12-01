import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ourhexenv import OurHexGame
from fhtw_hex.ppo_smaller import Agent
from fhtw_hex.random_agent import RandomAgent
from fhtw_hex.bit_smarter_agent import BitSmartAgent
import torch

def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    checkpoint = {
        'model_state_dict': agent.actor_critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_critic.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.actor_critic.critic_optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    try:
        checkpoint = torch.load(filename)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")

def train_against_agent(config):
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)
    
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        **config
    )
    
    for opponent, episodes in [("RandomAgent", 500), ("BitSmartAgent", 500)]:
        for episode in range(episodes):
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
                        action, probs, value = ppo_agent.choose_action(obs_flat, info)
                        env.step(action.item())
                        updated_reward = env.rewards[agent_id]
                        ppo_agent.remember(obs_flat, action, probs, value, updated_reward, done)
                        scores[agent_id] += updated_reward
                    elif agent_id == "player_2":
                        obs_flat = observation["observation"].flatten()
                        if opponent == "RandomAgent":
                            action = RandomAgent().select_action(env, info)
                        else:
                            action = BitSmartAgent().select_action(env, info)
                        env.step(action)
                        updated_reward = env.rewards[agent_id]
                        scores[agent_id] += updated_reward
                else:
                    env.step(None)
                terminations = env.terminations

            ppo_agent.learn()

    # Evaluate the agent
    eval_episodes = 100
    wins = 0
    for _ in range(eval_episodes):
        env.reset()
        done = False
        scores = {agent: 0 for agent in env.possible_agents}
        while not done:
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            if not done:
                if env.agent_selection == "player_1":
                    obs_flat = observation["observation"].flatten()
                    action, _, _ = ppo_agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[env.agent_selection]
                    scores[agent_id] += updated_reward
                else:
                    action = BitSmartAgent().select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[env.agent_selection]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)
        if scores["player_1"] > scores["player_2"]:
            wins += 1

    win_rate = wins / eval_episodes
    tune.track.log(win_rate=win_rate)

def main():
    ray.init(num_gpus=1)

    config = {
        "gamma": tune.uniform(0.9, 0.99),
        "actor_lr": tune.loguniform(1e-5, 1e-2),
        "critic_lr": tune.loguniform(1e-5, 1e-2),
        "gae_lambda": tune.uniform(0.9, 1.0),
        "policy_clip": tune.uniform(0.1, 0.3),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "n_epochs": tune.randint(5, 21)
    }

    scheduler = ASHAScheduler(
        max_t=1000,
        grace_period=100,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(train_against_agent, resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="win_rate",
            mode="max",
            scheduler=scheduler,
            num_samples=50
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_trial = results.get_best_trial("win_rate", "max", "last")
    print("Best trial config:", best_trial.config)
    print("Best trial final win rate:", best_trial.last_result["win_rate"])

    # Train final model with best hyperparameters
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)
    final_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        **best_trial.config
    )

    opponents = [
        (RandomAgent(), 'random', 1000),
        (BitSmartAgent(), 'bitsmart', 1000)
    ]

    for opponent_agent, suffix, episodes in opponents:
        print(f"\nTraining against {suffix.capitalize()}Agent...")
        train_against_agent(best_trial.config)
        save_ppo_checkpoint(final_agent, filename=f'ppo_checkpoint_after_{suffix}_tuned.pth')

    print("Training completed.")

if __name__ == "__main__":
    main()