class RandomAgent:

    def select_action(env, agent_id, info):
        action = env.action_space(agent_id).sample(info["action_mask"])
        return action