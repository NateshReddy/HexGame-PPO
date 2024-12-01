import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from fhtw_hex.ppo_memory import PPOBufferMemory
from fhtw_hex.acn import ActorCriticNetwork

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10, chkpt_dir="tmp/ppo", entropy_coef=0.01):
        self.gamma = gamma  # Diskontierungsfaktor
        self.policy_clip = policy_clip  # Clip-Wert für die PPO-Verlustfunktion
        self.n_epochs = n_epochs  # Anzahl der Trainingsdurchläufe
        self.gae_lambda = gae_lambda  # GAE-Lambda für Vorteilsschätzung
        self.chkpt_dir = chkpt_dir  # Initialisiere das Checkpoint-Verzeichnis
        self.entropy_coef = entropy_coef  # Koeffizient für den Entropy Bonus

        self.actor_critic = ActorCriticNetwork(n_actions, input_dims, alpha,
                                  chkpt_dir=chkpt_dir)  # Initialisierung des Actor-Netzwerks
        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic
        self.memory = PPOBufferMemory(batch_size)  # Initialisierung des Speichers

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

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor_critic.device)
        acts = self.actor(state)
        dist = Categorical(acts)
        value = self.critic(state)

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
        probs = T.tensor(probs[action]).to(self.actor_critic.device)

        value = T.squeeze(value).item()
        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.actor_critic.device)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)):
                advantage[t] = reward_arr[t] - values[t]
            advantage = T.tensor(advantage).to(self.actor_critic.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor_critic.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor_critic.device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(self.actor_critic.device)

                dist = self.actor(states)
                dist = Categorical(dist)
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
                self.actor_critic.actor_optimizer.zero_grad()
                self.actor_critic.critic_optimizer.zero_grad()

                total_loss.backward()
                
                self.actor_critic.actor_optimizer.step()
                self.actor_critic.critic_optimizer.step()

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
