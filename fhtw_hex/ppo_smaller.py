import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Klasse zur Verwaltung des Speichers für PPO (Proximal Policy Optimization)
class PPOMemory:
    def __init__(self, batch_size):
        # Initialisierung der Speicherlisten für Zustände, Aktionen, Wahrscheinlichkeiten, Werte, Belohnungen und 'done'-Flaggen
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size  # Größe der Batches

    # Methode zur Generierung von Batches aus den gespeicherten Erinnerungen
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    # Methode zum Speichern einer Erinnerung
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # Methode zum Löschen aller gespeicherten Erinnerungen
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# Neuronales Netzwerk für den Actor
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = f'{chkpt_dir}/actor_torch_ppo'
        os.makedirs(chkpt_dir, exist_ok=True)
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # Entpackt input_dims und übergibt es als Argument
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def update_checkpoint_dir(self, chkpt_dir):
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        os.makedirs(chkpt_dir, exist_ok=True)

    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Actor Network Architecture:\n")
                f.write(str(self.actor) + "\n")
                f.write("\nDevice: " + str(self.device) + "\n")
                f.write("\nParameters:\n")
                for name, param in self.named_parameters():
                    f.write(f"{name}: {param.shape}\n")


# Neuronales Netzwerk für den Kritiker
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = f'{chkpt_dir}/critic_torch_ppo'
        os.makedirs(chkpt_dir, exist_ok=True)
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # Entpackt input_dims und übergibt es als Argument
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def update_checkpoint_dir(self, chkpt_dir):
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        os.makedirs(chkpt_dir, exist_ok=True)

    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Critic Network Architecture:\n")
                f.write(str(self.critic) + "\n")
                f.write("\nDevice: " + str(self.device) + "\n")
                f.write("\nParameters:\n")
                for name, param in self.named_parameters():
                    f.write(f"{name}: {param.shape}\n")


# Agentenklasse, die den Actor und den Kritiker verwendet
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10, chkpt_dir="tmp/ppo", entropy_coef=0.01):
        self.gamma = gamma  # Diskontierungsfaktor
        self.policy_clip = policy_clip  # Clip-Wert für die PPO-Verlustfunktion
        self.n_epochs = n_epochs  # Anzahl der Trainingsdurchläufe
        self.gae_lambda = gae_lambda  # GAE-Lambda für Vorteilsschätzung
        self.chkpt_dir = chkpt_dir  # Initialisiere das Checkpoint-Verzeichnis
        self.entropy_coef = entropy_coef  # Koeffizient für den Entropy Bonus

        self.actor = ActorNetwork(n_actions, input_dims, alpha,
                                  chkpt_dir=chkpt_dir)  # Initialisierung des Actor-Netzwerks
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt_dir)  # Initialisierung des Kritiker-Netzwerks
        self.memory = PPOMemory(batch_size)  # Initialisierung des Speichers

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, allow_illegal_moves=False):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)

        if allow_illegal_moves:
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
        else:
            probs = dist.probs.detach().cpu().numpy().flatten()
            valid_actions = observation.flatten() == 0

            mask = valid_actions.astype(bool)
            # import ipdb; ipdb.set_trace()
            probs[~mask] = 0

            if np.sum(probs) > 0:
                probs /= np.sum(probs)
            else:
                probs[mask] = 1.0 / np.sum(mask)

            action = np.random.choice(len(probs), p=probs)
            probs = T.tensor(probs[action]).to(self.actor.device)

        value = T.squeeze(value).item()
        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.actor.device)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)):
                advantage[t] = reward_arr[t] - values[t]
            advantage = T.tensor(advantage).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[
                    batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # # Entropy regularization term to encourage exploration, comment in for experiment 5
                # entropy_bonus = dist.entropy().mean()
                # actor_loss -= self.entropy_coef * entropy_bonus

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def update_checkpoint_dir(self, chkpt_dir):
        self.chkpt_dir = chkpt_dir
        self.actor.update_checkpoint_dir(chkpt_dir)
        self.critic.update_checkpoint_dir(chkpt_dir)

    # Methode zum Drucken der Informationen des Agenten
    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Agent Information:\n")
                f.write("\nActor Network Info:\n")
            self.actor.print_info(file)
            with open(file, 'a') as f:
                f.write("\nCritic Network Info:\n")
            self.critic.print_info(file)
