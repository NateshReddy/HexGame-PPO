'''
Config File mit den man unterschiedliche Agents aufsetzen kann um Agent vs. Agent trainieren zu können.

'''

from fhtw_hex.ppo_torch import Agent
from fhtw_hex.utils import load_agent

####
# SET-UP für Agent 1
####

# Hyperparameter für Agent 1 
batch_size1 = 6  # Batch size
n_epochs1 = 2  # Number of epochs
alpha1 = 0.0005
n_actions1 = 7 * 7  # Assuming the board is 7x7
input_dims1 = [7 * 7]  # Flatten input

# Relative Pfade für unterschiedliche Agents die geladen werden sollen
# dir_agent1 = "fhtw_hex/experiment_5/agent1_12000"
dir_agent1 = "fhtw_hex/experiment_2/tmp/2024-06-29-02-50_smallPPO"

agent1 = load_agent(dir=dir_agent1,
                    n_actions=n_actions1,
                    input_dims=input_dims1,
                    batch_size=batch_size1,
                    n_epochs=n_epochs1,
                    alpha=alpha1,
                    gamma=0.99,
                    gae_lambda=0.95,
                    policy_clip=0.2)

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

####
# SET-UP für Agent 2
####

# Hyperparameter für Agent 2 
batch_size2 = 3  # Batch size
n_epochs2 = 2  # Number of epochs
alpha2 = 0.0005
n_games2 = 1000
n_actions2 = 7 * 7  # Assuming the board is 7x7
input_dims2 = [7 * 7]  # Flatten input
# dir_agent2 = "fhtw_hex/experiment_5/agent2_1200"
dir_agent2 = "fhtw_hex/experiment_2/tmp/2024-06-29-02-50_smallPPO"

agent2 = load_agent(dir=dir_agent2,
                    n_actions=n_actions2,
                    input_dims=input_dims2,
                    batch_size=batch_size2,
                    n_epochs=n_epochs2,
                    alpha=alpha2,
                    gamma=0.99,
                    gae_lambda=0.95,
                    policy_clip=0.2)
