import numpy as np
from fhtw_hex.ppo_torch import Agent
from fhtw_hex import hex_engine as engine
game = engine.hexPosition()  # Initialize the game
from fhtw_hex.utils import dir_setting, write_text_file, load_agent, get_last_folders, evaluate_pipeline
import fhtw_hex.experiment_6.experiment_6 as e6
import fhtw_hex.experiment_3.erperiment_3 as e3
import fhtw_hex.experiment_4.erperiment_4 as e4
import matplotlib.pyplot as plt

# Initialize hyperparameters: PPO Agent
board_size = 7
N = 12  # Frequency of learning
batch_size = 6  # Batch size
n_epochs = 2  # Number of epochs
alpha = 0.0005
n_games = 500
n_actions = board_size * board_size  # Assuming the board is 7x7
input_dims = [board_size * board_size]  # Flatten input
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2

# Initialize training runs
T = 15

# Model Trained
M = 2

def evaluate_all_agents(dirs, game, n_matches=10):
    n_agents = len(dirs)
    win_matrix = np.zeros((n_agents, n_agents))

    for i in range(n_agents):
        agent1 = load_agent(dirs[i], n_actions, input_dims, batch_size, n_epochs, alpha, gamma=gamma,
                            gae_lambda=gae_lambda, policy_clip=policy_clip)

        for j in range(n_agents):
            if i != j:
                agent2 = load_agent(dirs[j], n_actions, input_dims, batch_size, n_epochs, alpha, gamma=gamma,
                                    gae_lambda=gae_lambda, policy_clip=policy_clip)

                win_rate = evaluate_pipeline(game, agent1, agent2, n_matches=n_matches)
                win_matrix[i, j] = win_rate

    return win_matrix

def aggregate_and_print_win_rates(win_matrix):
    n_agents = win_matrix.shape[0]
    win_rates_as_agent1 = np.mean(win_matrix, axis=1)
    win_rates_as_agent2 = np.mean(1 - win_matrix, axis=0)

    for i in range(n_agents):
        print(f"Aggregate win rate of Agent {i+1} as Agent 1: {win_rates_as_agent1[i]:.2f}")
        print(f"Aggregate win rate of Agent {i+1} as Agent 2: {win_rates_as_agent2[i]:.2f}")

def plot_win_rates(win_matrix, agent_labels, figure_file):
    plt.figure(figsize=(10, 8))
    cax = plt.matshow(win_matrix, cmap='viridis')
    plt.colorbar(cax)
    plt.xticks(ticks=np.arange(len(agent_labels)), labels=agent_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(agent_labels)), labels=agent_labels)
    plt.xlabel('Opponent Agent')
    plt.ylabel('Evaluated Agent')
    plt.title('Win Rate Matrix for Agents')
    plt.savefig(figure_file + "win_rates.png")
    plt.show()

def pipline():
    test_rates = []  # Um zu evaluieren ob die win rate des neuen Agentens besser ist als die es alten Checkpoints
    count = 0
    done = False
    i = 0
    while i <= M:
        agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=gamma, alpha=alpha,
                      gae_lambda=gae_lambda, policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs)

        while not done:
            if count % 4 == 0:
                # Experiment Text
                txt = ("Agent gegen Random Pipeline Durchgang. Checkpoint eines Agent der für dieses Experiment gegen Random trainiert wurde")

                # EXPERIMENT 1: Gegen Random
                dir, figure_file = dir_setting("experiment_0")
                agent.update_checkpoint_dir(dir)
                write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)
                phase = count
                e6.train_vs_random(game, agent, 1, n_games, N, figure_file, str(phase) + "_random")

            else:
                # EXPERIMENT 1: Gegen vorangegangenen Checkpoint des Agents
                # Wichtig erst 2 Agent initialisieren bevor neuer Ordner erstellt wird
                dic = "fhtw_hex/experiment_0/tmp"
                last_folder = get_last_folders(dic, 1)
                agent_2 = load_agent(last_folder[0], n_actions, input_dims, batch_size, n_epochs, alpha, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip)

                dir, figure_file = dir_setting("experiment_0")
                agent.update_checkpoint_dir(dir)
                txt = ("Agent gegen alten Agent aus Pipeline Durchgang. Checkpoint eines Agent der für dieses Experiment gegen seinen Vorgänger trainiert wurde. Vorgänger Agent zu finden unter: " + dic)
                write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)
                agent_player = 1  # if count % 2 == 0 else -1  # Abwechselnd als Spieler 1 und Spieler -1
                phase = count
                e6.train_agent_vs_agent(game, agent, agent_player, agent_2, n_games, N, figure_file, str(phase) + "_agent")

            count += 1
            done = True if count == T else False

        # EVALUIEREN DES NEUEN AGENTS GEGENÜBER SEINEM DIREKTEN VORGÄNGER
        if len(test_rates) >= 2:
            dic = "fhtw_hex/final_agent/tmp"
            last_folder = get_last_folders(dic, T)
            win_matrix = evaluate_all_agents(last_folder, game, n_matches=10)
            aggregate_and_print_win_rates(win_matrix)

            dir, figure_file = dir_setting("experiment_0")
            plot_win_rates(win_matrix, [f"Agent {i+1}" for i in range(len(last_folder))], figure_file)

        else:
            test_rates.append(-np.inf)
            # SPEICHERN DES AGENTEN
            dir, _ = dir_setting("final_agent")
            agent.update_checkpoint_dir(dir)
            agent.save_models()
        i += 1

