from fhtw_hex.utils import plot_learning_curve
import numpy as np

from fhtw_hex.utils import plot_learning_curve
import numpy as np


def train_vs_random(game, agent, n_games, N, figure_file):
    best_score = -np.inf
    score_history = []
    avg_score = 0
    n_steps = 0
    for i in range(n_games):
        game.reset()
        done = False
        score = 0
        legal_moves_per_game = 0
        illegal_moves_per_game = 0
        learn_iters = 0
        agent_player = 1 if i / n_games == 2 else -1  # Abwechselnd als Spieler 1 und Spieler -1

        while not done:
            state = np.array(game.board).flatten()
            reward = 0

            if game.player == agent_player:
                action, prob, val = agent.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    legal_moves_per_game += 1
                    game.move(action_coordinates)
                else:
                    illegal_moves_per_game += 1
                    game.player *= -1

                if game.winner != 0:
                    print('Game winner:', game.winner)
                    if game.winner == agent_player:
                        reward = 1
                        reward -= legal_moves_per_game * 0.01  # Belohnung für weniger Züge
                    else:
                        reward = 0
                    done = True

                n_steps += 1
                agent.remember(state, action, prob, val, reward, done)

            else:
                game._random_move()
                if game.winner != 0:
                    print('Game winner:', game.winner)
                    reward = 0
                    done = True

            score += reward
            # if n_steps % N == 0:
            #     agent.learn()
            #     learn_iters += 1

        agent.learn()
        learn_iters += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Best Score: {best_score:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, '
            f'Illegal moves per game {illegal_moves_per_game} , Learning Steps per Game: {learn_iters}')

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    print("Training completed.")



###################################################################################################################################
'''

Agent vs. Agent training

'''
###################################################################################################################################
def agent_step(game, agent,agent_player, reward_adjustment):
    state = np.array(game.board).flatten()
    action, prob, val = agent.choose_action(state)
    action_coordinates = (action // 7, action % 7)
    legal_moves = 0
    illegal_moves = 0

    if action_coordinates in game.get_action_space():
        legal_moves += 1
        game.move(action_coordinates)
    else:
        illegal_moves += 1
        game.player *= -1

    reward = 0
    done = False

    if game.winner != 0:
        if game.winner == agent_player:
            reward = 1
            reward -= legal_moves * reward_adjustment
        else:
            reward = 0
        done = True

    agent.remember(state, action, prob, val, reward, done)
    agent.learn()

    return reward, done, legal_moves, illegal_moves

def update_scores_and_save_models(score, score_history, best_score, agent):
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    return avg_score, best_score

def train_agent_vs_agent(game, agent_one, agent_player, agent_two, n_games, N, figure_file):
    best_score_one = -np.inf
    best_score_two = -np.inf
    score_history_one = []
    avg_score_one = 0
    n_steps = 0

    for i in range(n_games):
        game.reset()
        done = False
        score_one = 0
        legal_moves_per_game = 0
        illegal_moves_per_game = 0
        learn_iters_one = 0

        while not done:
            if game.player == agent_player:
                reward, done, legal_moves, illegal_moves = agent_step(game, agent_one,agent_player, 0.01)
                score_one += reward
                learn_iters_one += 1
                legal_moves_per_game += legal_moves
                illegal_moves_per_game += illegal_moves
            else:
                state = np.array(game.board).flatten()
                action, _, _ = agent_two.choose_action(state)
                action_coordinates = (action // 7, action % 7)
                if action_coordinates in game.get_action_space():
                    game.move(action_coordinates)
                if game.winner != 0:
                    if game.winner == agent_player:
                        reward = 1
                        # reward -= legal_moves * reward_adjustment
                    else:
                        reward = 0
                    done = True

            n_steps += 1

        avg_score_one, best_score_one = update_scores_and_save_models(score_one, score_history_one, best_score_one, agent_one)

        print(
            f'Episode {i}, Agent One Score: {score_one:.2f}, Agent One Avg Score: {avg_score_one:.2f}, Best Score One: {best_score_one:.2f}, mBest Score Two: {best_score_two:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, Illegal moves per game {illegal_moves_per_game}, Learning Steps per Game Agent One: {learn_iters_one}')

    x = [i + 1 for i in range(len(score_history_one))]
    plot_learning_curve(x, score_history_one, figure_file)
    print("Training completed.")
