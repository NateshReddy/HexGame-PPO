import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ppo_hex.ppo_memory import PPOBufferMemory
from ppo_hex.acn import ActorCriticNetwork

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, actor_lr=0.0003, critic_lr=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10, chkpt_dir="tmp/ppo", entropy_coef=0.01):
        
        # Initialize hyperparameters for PPO
        self.gamma = gamma  # Discount factor for future rewards
        self.policy_clip = policy_clip  # Clip value for PPO loss function
        self.n_epochs = n_epochs  # Number of training epochs per update
        self.gae_lambda = gae_lambda  # GAE lambda for advantage estimation
        self.chkpt_dir = chkpt_dir  # Directory to save checkpoints
        self.entropy_coef = entropy_coef  # Coefficient for entropy bonus

        # Initialize the actor-critic network
        self.actor_critic = ActorCriticNetwork(n_actions, input_dims, actor_lr, critic_lr,
                                               chkpt_dir=chkpt_dir)
        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic

        # Initialize memory buffer for storing experiences
        self.memory = PPOBufferMemory(batch_size)
        
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.actor_critic.to(self.device)

    def remember(self, state, action, probs, vals, reward, done):
        # Store experience in memory buffer
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        # Save model checkpoints for actor and critic networks
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        # Load model checkpoints for actor and critic networks
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, info):
        # Convert observation to tensor and move to device
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.device)
        
        # Use the actor-critic network to choose an action based on the current state and action mask
        return self.actor_critic.act(state, info["action_mask"])

    # def calculate_gae(self, rewards, values, dones):
    #     advantages_arr = []
    #     for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
    #         advantages = []
    #         last_advantage = 0

    #         for t in reversed(range(len(ep_rews))):
    #             if t + 1 < len(ep_rews):
    #                 delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
    #             else:
    #                 delta = ep_rews[t] - ep_vals[t]

    #             advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
    #             last_advantage = advantage
    #             advantages.insert(0, advantage)

    #         advantages_arr.extend(advantages)

    #     return torch.tensor(advantages_arr, dtype=torch.float)
       
    def learn(self):
        # Reset action counts (for exploration strategies like UCB)
        self.actor_critic.reset_action_counts()
        
        for _ in range(self.n_epochs):
            # Generate batches from memory buffer
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.device)  # Convert value estimates to tensor on device
            
            advantage = np.zeros(len(reward_arr), dtype=np.float32)  # Initialize advantage array

            for t in range(len(reward_arr)):
                advantage[t] = reward_arr[t] - values[t]  # Calculate advantage

            advantage = T.tensor(advantage).to(self.device)  # Convert advantage to tensor on device

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(self.device)

                critic_value, new_probs, entropy = self.actor_critic.evaluate(states, actions)

                critic_value = T.squeeze(critic_value)  # Remove dimensions of size 1

                prob_ratio = new_probs.exp() / old_probs.exp()  # Calculate probability ratio for PPO
                
                weighted_probs = advantage[batch] * prob_ratio  # Calculate weighted probabilities
                
                weighted_clipped_probs = T.clamp(prob_ratio,
                                                 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]  # Apply clipping
                
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()  # Calculate actor loss with clipping

                returns = advantage[batch] + values[batch]  # Calculate returns
                
                critic_loss = (returns - critic_value) ** 2  # Calculate critic loss (MSE)
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss  # Total loss is a combination of actor and critic loss
                
                self.actor_critic.actor_optimizer.zero_grad()  # Zero gradients for optimizer
                self.actor_critic.critic_optimizer.zero_grad()

                total_loss.backward()  # Backpropagate total loss
                
                self.actor_critic.actor_optimizer.step()  # Update actor network weights
                self.actor_critic.critic_optimizer.step()  # Update critic network weights

        self.memory.clear_memory()  # Clear memory buffer after learning

    '''
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.device)
            
            # Calculate GAE
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            last_gae = 0
            for t in reversed(range(len(reward_arr) - 1)):
                if dones_arr[t]:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                delta = reward_arr[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae * (1 - dones_arr[t])
                advantages[t] = last_gae

            advantages = T.tensor(advantages).to(self.device)
            returns = advantages + values

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(self.device)

                critic_value, new_probs, entropy = self.actor_critic.evaluate(states, actions)
                critic_value = T.squeeze(critic_value)

                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Entropy regularization term (optional)
                actor_loss -= self.entropy_coef * entropy

                critic_loss = nn.functional.mse_loss(critic_value, returns)
                
                # total_loss = actor_loss + 0.5 * critic_loss
                self.actor_critic.actor_optimizer.zero_grad()
                self.actor_critic.critic_optimizer.zero_grad()

                actor_loss.backward()
                critic_loss.backward()
                
                self.actor_critic.actor_optimizer.step()
                self.actor_critic.critic_optimizer.step()

        self.memory.clear_memory()
    '''

    def update_checkpoint_dir(self, chkpt_dir):
        self.chkpt_dir = chkpt_dir
        
        # Update checkpoint directory for both actor and critic networks
        self.actor.update_checkpoint_dir(chkpt_dir)
        self.critic.update_checkpoint_dir(chkpt_dir)

    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Agent Information:\n")
                f.write("\nActor Network Info:\n")
            self.actor.print_info(file)
            with open(file, 'a') as f:
                f.write("\nCritic Network Info:\n")
            self.critic.print_info(file)

    def select_action(self, observation, reward, termination, truncation, info):
        action, _, _ = self.choose_action(observation["observation"].flatten(), info)
        return action.item()

    @classmethod
    def from_file(cls, filename, env):
        agent = cls(
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
        checkpoint = T.load(filename, map_location=T.device('cpu'))
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        agent.actor_critic.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.actor_critic.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return agent