import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(ActorCriticNetwork, self).__init__()

        self.actor_checkpoint_file = f'{chkpt_dir}/actor_ppo'
        self.critic_checkpoint_file = f'{chkpt_dir}/critic_ppo'
        os.makedirs(chkpt_dir, exist_ok=True)

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1)
        )

        self.actor_optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

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
        self.actor_checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.critic_checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
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