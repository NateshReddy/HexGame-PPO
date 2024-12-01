import numpy as np

class RandomAgent:

    def select_action(self, env, info):
        available_moves = [i for i, valid in enumerate(info["action_mask"]) if valid]
        return np.random(available_moves)