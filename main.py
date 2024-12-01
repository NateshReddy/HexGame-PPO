import config
import numpy as np
import os
from fhtw_hex import hex_engine as engine
game = engine.hexPosition() # Initialize the game
# from fhtw_hex.experiment_5.ppo_entropy_bonus import Agent   # das ist schlecht beim lernen nur ca. 55%
from fhtw_hex.ppo_torch import Agent
from fhtw_hex.ppo_smaller import Agent as Agent6
from fhtw_hex.utils import dir_setting, write_text_file, load_agent
import fhtw_hex.experiment_1.experiment_1 as e1
import fhtw_hex.experiment_2.experiment_2_phasenweise as e2
import fhtw_hex.experiment_3.erperiment_3 as e3
import fhtw_hex.experiment_4.erperiment_4 as e4
import fhtw_hex.experiment_5.experiment_5_abwechselnd as e5
from fhtw_hex.utils import evaluate

# from pipline import pipline

# if __name__ == "__main__":
## pipeline is experiment 6
#     pipline()
#     print("")
#     print("")
#     print("")
#     print("___________________________Finish______________________________")





#########################################################################################################################################################################################
def experiment_1():
    # Initialize parameters
    N = 6  # Frequency of learning
    batch_size = 3  # Batch size
    n_epochs = 2  # Number of epochs
    alpha = 0.0005
    n_games = 2000
    n_actions = 7 * 7  # Assuming the board is 7x7
    input_dims = [7 * 7]  # Flatten input

    dir, figure_file = dir_setting("experiment_1")

    txt = ("In diesem Experiment initalisieren wir einen Agenten der für Spieler Weiß (Spieler 1) oder Schwarz (-1) zieht und dabei trainiert. "
           "Der Gegenzug wird hierbei von Random gezogen.")

    # Initialize the PPO agent
    agent = Agent6(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)

    # Write the setup file
    write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)

    # Train the agent
    e1.train(game, agent, n_games, N, figure_file)
#####################################################################################################################################################################################
def experiment_2():
    # Initialize parameters
    N = 6  # Frequency of learning
    batch_size = 3  # Batch size
    n_epochs = 2  # Number of epochs
    alpha = 0.0005
    n_games = 4000
    n_actions = 7 * 7  # Assuming the board is 7x7
    input_dims = [7 * 7]  # Flatten input
    dir, figure_file = dir_setting("experiment_2")
    txt = ("In diesem Experiment initalisieren wir einen Agenten der dann abwechselnd für Spieler Weiß (Spieler 1) und Spieler Schwarz (Spieler 2) zieht und dabei trainiert. "
           "Der Gegenzug wird hierbei abwechselnd von Random gezogen.")
    # Initialize the PPO agent
    agent = Agent6(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)
    # Write the setup file
    write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)
    # Train the agent
    e2.train(game, agent, n_games, N, figure_file)


#####################################################################################################################################################################################
def experiment_3():
    N = 16  # Frequency of learning
    batch_size = 4  # Batch size
    n_epochs = 4  # Number of epochs
    alpha = 1e-3
    n_games = 1000
    # Initialize the PPO agent
    n_actions = 7 * 7  # Assuming the board is 7x7
    input_dims = [7 * 7]  # Flatten input
    dir, figure_file = dir_setting("experiment_3")
    agent1 = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha, 
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)
    agent2 = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha, 
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)
    txt = ("Agent spielt die Hälfte der Zeit als Weiß gegen Random und dann als schwarz gegen Random")
    # Write the setup file
    write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)
    e3.train(game, agent1, n_games, N, figure_file)
    e3.train(game, agent2, n_games, N, figure_file)


    # Zweiter teil des Experiments
    def get_last_two_folders(directory):
        # Liste alle Einträge im Verzeichnis auf und filtere nur die Ordner heraus
        all_entries = os.listdir(directory)
        all_folders = [entry for entry in all_entries if os.path.isdir(os.path.join(directory, entry))]

        # Sortiere die Ordner nach ihrem Erstellungsdatum oder alphabetisch
        all_folders.sort(key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))

        # Greife auf die letzten beiden Ordner zu
        if len(all_folders) >= 2:
            last_folder = all_folders[-2]
            second_last_folder = all_folders[-1]
            return os.path.join(directory, last_folder), os.path.join(directory, second_last_folder)
        else:
            raise ValueError("Das Verzeichnis enthält weniger als zwei Ordner.")

    # Beispielverwendung
    directory_path = 'fhtw_hex/experiment_3/tmp'
    last_two_folders = get_last_two_folders(directory_path)

    dir1 = directory_path+last_two_folders[0]
    agent1 = load_agent(dir1, n_actions, input_dims, batch_size, n_epochs, alpha, gamma=0.99, gae_lambda= 0.95, policy_clip=0.2)

   
    dir2 = directory_path+last_two_folders[1]
    agent2 = load_agent(dir2, n_actions, input_dims, batch_size, n_epochs, alpha, gamma=0.99, gae_lambda= 0.95, policy_clip=0.2)

    e3.train_agent_vs_agent(game, agent1,agent2,N,figure_file)


#####################################################################################################################################################################################
def experiment_4():
    N = 6  # Frequency of learning
    batch_size = 3  # Batch size
    n_epochs = 2  # Number of epochs
    alpha = 1e-3
    n_games = 7000
    # Initialize the PPO agent
    n_actions = 7 * 7  # Assuming the board is 7x7
    input_dims = [7 * 7]  # Flatten input
    dir, figure_file = dir_setting("experiment_4")
    agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.95, alpha=alpha, 
                  gae_lambda=0.90, policy_clip=0.3, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=dir)
    txt = ("Bei diesem Experiment startet der Agent mit einem Modifizierten Spielbrett um dieses zu verkleinern. So sind am Anfang des Training jeweils Reihen schon mit 1 und -1 vor ausgefüllt, sodass das Spielfeld, in dem tatsächliche Züge ausgeführt werden können kleiner ist (die Seiten die bespielt werden müssen, damit 1 gewinnt sind hierbei schon mit 1 ausgefüllt und die seiten die ")
    # Write the setup file
    write_text_file(dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)
    e4.train(game, agent, n_games, N, figure_file)

#####################################################################################################################################################################################
def experiment_5():
    # Initialize parameters
    N = 6  # Frequency of learning
    batch_size = 3  # Batch size
    n_epochs = 2  # Number of epochs
    alpha = 0.0003
    n_games = 12000
    n_actions = 7 * 7  # Assuming the board is 7x7
    input_dims = [7 * 7]  # Flatten input

    txt_dir, figure_file = dir_setting("experiment_5")

    txt = ("In diesem Experiment trainieren zwei Agents mit self play phasenweise.")
    agent1_dir = "fhtw_hex/experiment_5/agent1"
    agent2_dir = "fhtw_hex/experiment_5/agent2"

    # Initialize the PPO agent
    agent1 = Agent6(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=agent1_dir)
    agent2 = Agent6(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                  gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=agent2_dir)
    agent2.print_info(txt_dir + " model Summary")

    # Write the setup file
    write_text_file(txt_dir, txt, N=N, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha, n_games=n_games, n_actions=n_actions, input_dims=input_dims)

    # Train the agent
    ag1, ag2 = e5.train(game, agent1, agent2, n_games, N, figure_file)

    print('\n Evaluate the agents')
    win_rate_agent1 = evaluate(game, ag1)
    print(f'Win Rate of Agent1 against Random: {win_rate_agent1:.2f}')
    win_rate_agent2 = evaluate(game, ag2)
    print(f'Win Rate of Agent2 against Random: {win_rate_agent2:.2f}')


def experimence():
    # experiment_1()
    experiment_2()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()

    
#####################################################################################################################################################################################
def play():
    # geladen aus den Cofig File 
    agent1 = config.agent1
    agent2 = config.agent2
    def agent_move(board, action_set, agent):
        state = np.array(board).flatten()
        action, _, _ = agent.choose_action(state)
        return (action // 7, action % 7)
    print("Choose play mode:")
    print("1: Human vs Machine")
    print("2: Machine vs Machine")
    print("3: Machine vs Random")
    play_mode = int(input("Enter the number of the desired play mode: "))
    print("Choose your side:")
    print("1: Play as Player 1")
    print("2: Play as Player 2")
    player_side = int(input("Enter the number of your side: "))
    if play_mode == 1:
        if player_side == 1:
            # human is player 1
            game.human_vs_machine(machine=lambda board, action_set: agent_move(board, action_set, agent2))
        else:
            game.human_vs_machine(human_player=-1, machine=lambda board, action_set: agent_move(board, action_set, agent1)) #agent sollte auswählbar sein
    elif play_mode == 2:
        game.machine_vs_machine(machine1=lambda board, action_set: agent_move(board, action_set, agent1),
                                machine2=lambda board, action_set: agent_move(board, action_set, agent2))
    elif play_mode == 3:
        if player_side == 1:
            game.machine_vs_machine(machine1=lambda board, action_set: agent_move(board, action_set, agent1))
        else:
            game.machine_vs_machine(machine2=lambda board, action_set: agent_move(board, action_set, agent2))
    else:
        print("Invalid selection")


#####################################################################################################################################################################################
if __name__ == "__main__":
    mode = input("Enter 'train' to train the agents or 'play' to play the game: ").strip().lower()

    if mode == 'train':
        experimence()

    elif mode == 'play':
        play()

    else:
        print("Invalid mode selected. Please enter 'train' or 'play'.")


