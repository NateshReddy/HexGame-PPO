from fhtw_hex.utils import plot_learning_curve
import numpy as np


def train(game, agent, n_games, N, figure_file):
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
        agent_player = 1 if i % 2 == 0 else -1  # Abwechselnd als Spieler 1 und Spieler -1

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
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        agent.learn()

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

