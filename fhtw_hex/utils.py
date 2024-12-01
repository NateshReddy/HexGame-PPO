from fhtw_hex.ppo_smaller import Agent as Agent6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import datetime
import random


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg, label=f'Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    # Set y-ticks to include the highest score with formatting
    max_score = max(scores)
    min_score = min(scores)
    plt.ylim(bottom=min_score, top=max_score)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.savefig(figure_file, bbox_inches='tight')
    plt.close()

def plot_win_history(x, history, phase, figure_file):
    plt.show()
    running_avg = np.zeros(len(history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(history[max(0, i - 100):(i + 1)])
    plt.plot(x[99:], running_avg[99:], label=phase)
    plt.title(f'Running average of previous {len(history)} games')
    plt.legend()
    plt.savefig(figure_file+"win_history.png")

def plot_learning_curve_agents(x, scores_agent1, scores_agent2, figure_file):
    running_avg_agent1 = np.zeros(len(scores_agent1))
    running_avg_agent2 = np.zeros(len(scores_agent2))

    for i in range(len(running_avg_agent1)):
        running_avg_agent1[i] = np.mean(scores_agent1[max(0, i - 100):(i + 1)])
        running_avg_agent2[i] = np.mean(scores_agent2[max(0, i - 100):(i + 1)])

    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg_agent1, label='Agent1')
    plt.plot(x, running_avg_agent2, label='Agent2')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    # Set y-ticks to include the highest score
    if max(scores_agent1) > max(scores_agent2):
        max_score = max(scores_agent1)
    else:
        max_score = max(scores_agent2)
    plt.ylim(bottom=0, top=max_score+5)
    plt.yticks(np.linspace(0, max_score, num=11))  # 11 ticks from 0 to max_score
    plt.savefig(figure_file, bbox_inches='tight')
    plt.close()

# Methode um Pfade zum speichern von Modellen zu setzen
def dir_setting(experiment_dir: str):
    # pattern = re.compile(r'^experiment(_\d+)?$')
    # if not pattern.match(experiment_dir):
    #     raise ValueError(
    #         f"Invalid experiment_dir format: {experiment_dir}. Expected format is 'experiment_2', or 'experiment_nn' where nn is a placeholder for numbers.")
    dir = f"fhtw_hex/{experiment_dir}/tmp/{str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))}"
    figure_file = dir + '/plots/hex_learning_curve.png'
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    return dir, figure_file


# Methode um Agents zu laden
def load_agent(dir, n_actions, input_dims, batch_size, n_epochs, alpha, gamma=0.99, gae_lambda=0.95, policy_clip=0.2):
    agent = Agent6(n_actions=n_actions, input_dims=input_dims, gamma=gamma, alpha=alpha, gae_lambda=gae_lambda,
                  policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)
    agent.load_models()  # Lade die Modelle
    return agent


def write_text_file(directory, text, **params):
    """
    Schreibt eine .txt Datei in einen bestimmten Ordner mit zusätzlichen Parametern.

    :param directory: Der Ordner, in dem die Datei erstellt werden soll.
    :param text: Der Text, der in die Datei geschrieben werden soll.
    :param params: Zusätzliche Parameter, die ebenfalls in die Datei geschrieben werden.
    """
    # Sicherstellen, dass das Verzeichnis existiert
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Pfad zur Datei erstellen
    file_path = os.path.join(directory, "Setup.txt")

    # Datei schreiben
    with open(file_path, 'w') as file:
        # Text schreiben
        file.write(f"{text}\n")

        # Zusätzliche Parameter schreiben
        for key, value in params.items():
            file.write(f"{key}: {value}\n")


def evaluate(game, agent, n_matches=20):
    """
        Evaluates the performance of a given agent by playing a specified number of matches against a random player.

        Parameters:
        game (hexPosition): An instance of the hexPosition game.
        agent: The agent to be evaluated.
        n_matches (int): The number of matches to be played for evaluation.

        Returns:
        float: The win rate of the agent against the random player.

        Additionally, prints the list of starting players and the list of winners for each match.
    """
    win_count_agent = 0
    winners = []
    starting_players = []

    for _ in range(n_matches):
        game.reset()
        done = False
        starting_player = 1 if random.choice([True, False]) else -1
        starting_players.append("Agent" if starting_player == 1 else "Random")
        game.player = starting_player

        while not done:
            state = np.array(game.board).flatten()
            if game.player == 1:
                action, _, _ = agent.choose_action(state)
            else:
                action = random.choice(game.get_action_space())
                action = action[0] * len(game.board) + action[1]  # Convert to scalar

            action_coordinates = (action // len(game.board), action % len(game.board))
            game.move(action_coordinates)

            if game.winner != 0:
                winners.append("Agent" if (game.winner == 1 and starting_player == 1) or (
                            game.winner == -1 and starting_player == -1) else "Random")
                if game.winner == 1 and starting_player == 1:
                    win_count_agent += 1
                elif game.winner == -1 and starting_player == -1:
                    win_count_agent += 1
                done = True

    win_rate = win_count_agent / n_matches
    print(f'Starting Players: {starting_players}')
    print(f'Winners: {winners}')
    return win_rate


def evaluate_pipeline(game, new_agent, old_agent, n_matches=10):
    win_count_agent = 0
    winners = []
    starting_players = []
    half_games = int(n_matches/2) # nur ganze zahlen bitte

    #Erste Hälfte spielt er als Weiß
    for _ in range(half_games):
        game.reset()
        done = False
        while not done:
            state = np.array(game.board).flatten()
            if game.player == 1:
                action, _, _ = new_agent.choose_action(state)
            else:
                action, _, _ = old_agent.choose_action(state)
            action_coordinates = (action // len(game.board), action % len(game.board))
            game.move(action_coordinates)

            if game.winner != 0:
                winners.append("New Agent" if (game.winner == 1) else "Old Agent")
                if game.winner == 1:
                    win_count_agent += 1
                done = True

    # Zweite Hälfte spielt er als Schwarz
    for _ in range(half_games):
        game.reset()
        done = False
        while not done:
            state = np.array(game.board).flatten()
            if game.player == -1:
                action, _, _ = new_agent.choose_action(state)
            else:
                action, _, _ = old_agent.choose_action(state)
            action_coordinates = (action // len(game.board), action % len(game.board))
            game.move(action_coordinates)

            if game.winner != 0:
                winners.append("New Agent" if (game.winner == -1) else "Old Agent")
                if game.winner == -1:
                    win_count_agent += 1
                done = True
    win_rate = win_count_agent / n_matches
    print(f'Starting Players: {starting_players}')
    print(f'Winners: {winners}')
    return win_rate



def get_last_folders(directory, amount):
    # Liste alle Einträge im Verzeichnis auf und filtere nur die Ordner heraus
    all_entries = os.listdir(directory)
    all_folders = [entry for entry in all_entries if os.path.isdir(os.path.join(directory, entry))]

    # Sortiere die Ordner nach ihrem Erstellungsdatum oder alphabetisch
    all_folders.sort(key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))

    # Greife auf die letzten 'amount' Ordner zu
    if len(all_folders) >= amount:
        last_folders = all_folders[-amount:]  # Nimm die letzten 'amount' Ordner
        last_folders_paths = [os.path.join(directory, folder) for folder in last_folders]
        
        # Ersetze Backslashes durch Schrägstriche in den Pfaden
        last_folders_paths = [path.replace('\\', '/') for path in last_folders_paths]
        
        return last_folders_paths
    else:
        raise ValueError(f"Das Verzeichnis enthält weniger als {amount} Ordner.")