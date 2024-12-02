import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# class CategoricalMasked(Categorical):
#     """Masked categorical distribution for action selection."""
#     def __init__(self, probs: T.Tensor, mask: T.Tensor):
#         self.batch, self.nb_action = probs.size()
#         self.tensor_mask = T.stack([mask.bool()]*self.batch, dim=0)
#         self.all_zeros = T.zeros_like(probs)
#         masked_probs = T.where(self.tensor_mask, probs, self.all_zeros)
#         # Add epsilon to avoid division by zero
#         #masked_probs += 1e-8
        
#         # Normalize probabilities
#         masked_probs /= masked_probs.sum(dim=-1, keepdim=True)
#         #normalized_probs = (self.all_zeros + 1e-8) / (self.all_zeros + 1e-8).sum(dim=-1, keepdim=True)
#         super().__init__(probs=masked_probs)

import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, actor_alpha, critic_alpha,
                 fc1_dims=512, fc2_dims=512, fc3_dims=256, fc4_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorCriticNetwork, self).__init__()

        self.actor_checkpoint_file = f'{chkpt_dir}/actor_ppo'
        self.critic_checkpoint_file = f'{chkpt_dir}/critic_ppo'
        os.makedirs(chkpt_dir, exist_ok=True)

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, 1)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_alpha)
        self.device = T.device('cpu')
        self.to(self.device)


class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, actor_lr, critic_lr, fc1_dims=512, fc2_dims=512, fc3_dims=256, fc4_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorCriticNetwork, self).__init__()
        self.action_counts = np.zeros(n_actions)

        self.actor_checkpoint_file = f'{chkpt_dir}/actor_ppo'
        self.critic_checkpoint_file = f'{chkpt_dir}/critic_ppo'
        os.makedirs(chkpt_dir, exist_ok=True)

        # self.actor = nn.Sequential(
        #     nn.Linear(*input_dims, fc1_dims),
        #     nn.Tanh(),
        #     nn.Linear(fc1_dims, fc2_dims),
        #     nn.Tanh(),
        #     nn.Linear(fc2_dims, n_actions),
        #     nn.Softmax(dim=-1)
        # )

        # self.critic = nn.Sequential(
        #     nn.Linear(*input_dims, fc1_dims),
        #     nn.Tanh(),
        #     nn.Linear(fc1_dims, fc2_dims),
        #     nn.Tanh(),
        #     nn.Linear(fc2_dims, 1)
        # )

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, 1)
        )

        self.actor_optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"

    def forward(self, state):
        dist = self.actor(state)
        value = self.critic(state)
        return dist, value
    
    def save_checkpoint(self):
        T.save(self.actor.state_dict(), self.actor_checkpoint_file)
        T.save(self.critic.state_dict(), self.critic_checkpoint_file)

    def load_checkpoint(self):
        self.actor.load_state_dict(T.load(self.actor_checkpoint_file))
        self.critic.load_state_dict(T.load(self.critic_checkpoint_file))

    def update_checkpoint_dir(self, chkpt_dir):
        self.actor_checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.critic_checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        os.makedirs(chkpt_dir, exist_ok=True)

    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Actor-Critic Network Architecture:\n")
                f.write("Actor:\n")
                f.write(str(self.actor) + "\n")
                f.write("\nCritic:\n")
                f.write(str(self.critic) + "\n")
                f.write("\nDevice: " + str(self.device) + "\n")
                f.write("\nParameters:\n")
                for name, param in self.named_parameters():
                    f.write(f"{name}: {param.shape}\n")
    
    '''
    def act(self, state: np.ndarray, mask: np.ndarray):
        """Compute action and log probabilities with UCB exploration."""
        # Actor newtork outputs the probability for each action
        probs = self.actor(state)
        probs_np = probs.detach().cpu().numpy().flatten()

        # Apply mask to filter invalid actions
        valid_actions = mask.astype(bool)
        probs_np[~valid_actions] = 0

        # Normalize probabilities
        if np.sum(probs_np) > 0:
            probs_np /= np.sum(probs_np)
        else:
            probs_np[valid_actions] = 1.0 / np.sum(valid_actions)

        # UCB calculation
        t = np.sum(self.action_counts) + 1
        ucb_values = probs_np + np.sqrt(2 * np.log(t) / (self.action_counts + 1e-5))
        ucb_values[~valid_actions] = -np.inf  # Ensure invalid actions are not selected

        # Choose action based on UCB - select action with maximum UCB value
        action = np.argmax(ucb_values)

        # Update count for selected action
        self.action_counts[action] += 1

        # Convert back to tensor
        action_tensor = T.tensor([action], dtype=T.long).to(self.device)
        probs_tensor = T.tensor([probs_np[action]], dtype=T.float).to(self.device)

        # Compute log probability and state value
        action_log_prob = T.log(probs_tensor)
        state_val = self.critic(state)
        return action_tensor.detach(), action_log_prob.detach(), state_val.detach()
    '''

    def act(self, state: np.ndarray, mask: np.ndarray):
        """Compute action and log probabilities."""
        probs = self.actor(state)
        # Convert probs to numpy array
        probs_np = probs.detach().cpu().numpy().flatten()

        # Apply mask
        valid_actions = mask.astype(bool)
        probs_np[~valid_actions] = 0

        # Normalize probabilities
        if np.sum(probs_np) > 0:
            probs_np /= np.sum(probs_np)
        else:
            probs_np[valid_actions] = 1.0 / np.sum(valid_actions)

        # Choose action
        action = np.random.choice(len(probs_np), p=probs_np)

        # Convert back to tensor
        action_tensor = T.tensor([action], dtype=T.long).to(self.device)
        probs_tensor = T.tensor([probs_np[action]], dtype=T.float).to(self.device)

        # Compute log probability and state value
        action_log_prob = T.log(probs_tensor)
        state_val = self.critic(state)
        return action_tensor.detach(), action_log_prob.detach(), state_val.detach()
    
    def evaluate(self, state: np.ndarray, action: T.Tensor):
        """Evaluate state for critic value."""
        probs = self.actor(state)
        distribution = T.distributions.Categorical(probs)
        action_log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        state_val = self.critic(state)
        return state_val, action_log_prob, entropy

    def reset_action_counts(self):
        self.action_counts = np.zeros_like(self.action_counts)
