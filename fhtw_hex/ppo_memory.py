import numpy as np
import torch

class PPOBufferMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size  

    def generate_batches(self):
        n_states = len(self.states)
        
        if n_states == 0:
            return None, None, None, None, None, None, []  # Return empty if no states

        # Create indices and shuffle them
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        # Create batches
        batches = [indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        # Convert lists to NumPy arrays (move CUDA tensors to CPU if necessary)
        return (
            np.array([state.cpu().numpy() if isinstance(state, torch.Tensor) and state.is_cuda else state for state in self.states]),
            np.array([action.cpu().numpy() if isinstance(action, torch.Tensor) and action.is_cuda else action for action in self.actions]),
            np.array([prob.cpu().numpy() if isinstance(prob, torch.Tensor) and prob.is_cuda else prob for prob in self.probs]),
            np.array([val.cpu().numpy() if isinstance(val, torch.Tensor) and val.is_cuda else val for val in self.vals]),
            np.array([reward.cpu().numpy() if isinstance(reward, torch.Tensor) and reward.is_cuda else reward for reward in self.rewards]),
            np.array([done.cpu().numpy() if isinstance(done, torch.Tensor) and done.is_cuda else done for done in self.dones]),
            batches
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.probs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.vals.clear()