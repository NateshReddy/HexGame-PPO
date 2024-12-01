def will_opponent_win_next(game, player, min_stones=6):
    opponent = -player
    size = len(game.board)

    # Prüfen, ob der Gegner genügend Steine gelegt hat
    opponent_stones = sum(row.count(opponent) for row in game.board)
    if opponent_stones < min_stones:
        return None

    for r in range(size):
        for c in range(size):
            if game.board[r][c] == 0:  # Leeres Feld
                game.board[r][c] = opponent
                if opponent == 1:
                    has_won, _ = game._evaluate_white(False)
                else:
                    has_won, _ = game._evaluate_black(False)
                game.board[r][c] = 0
                game.winner = 0 #setzt winner auf 0 wegen evaluate_methoden
                if has_won:
                    return (r, c)
    return None


def is_blocking_move(board, action_coordinates, player):
    opponent = -player
    row, col = action_coordinates
    size = len(board)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

    for dr, dc in directions:
        r1, c1 = row + dr, col + dc
        r2, c2 = row - dr, col - dc
        if (0 <= r1 < size and 0 <= c1 < size and board[r1][c1] == opponent) and \
                (0 <= r2 < size and 0 <= c2 < size and board[r2][c2] == opponent):
            return True
    return False

def is_connection_blocking_move(board, action_coordinates, player):
    opponent = -player
    row, col = action_coordinates

    # Define the main axes directions based on the updated clarification
    main_axes = [
        [(0, -1), (0, 1)],  # Vertical axis: up and down
        [(-1, 0), (1, 0)],  # Horizontal axis: left and right
        [(1, -1), (-1, 1)]  # Diagonal axis: down-left and up-right
    ]

    for direction in main_axes:
        (dr1, dc1), (dr2, dc2) = direction

        # Coordinates for checking opponent's stones
        r1, c1 = row + dr1, col + dc1
        r2, c2 = row + dr2, col + dc2

        # Check if both coordinates are within board bounds
        if (0 <= r1 < len(board) and 0 <= c1 < len(board[0])) and (
                0 <= r2 < len(board) and 0 <= c2 < len(board[0])):
            # Check if both surrounding positions contain opponent's stones
            if board[r1][c1] == opponent and board[r2][c2] == opponent:
                # Ensure the middle position (action_coordinates) is empty
                return True
    return False


def is_connecting_move(board, action_coordinates, player):
    row, col = action_coordinates
    size = len(board)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

    for dr, dc in directions:
        r1, c1 = row + dr, col + dc
        r2, c2 = row - dr, col - dc
        if (0 <= r1 < size and 0 <= c1 < size and board[r1][c1] == player) or \
                (0 <= r2 < size and 0 <= c2 < size and board[r2][c2] == player):
            return True
    return False


def has_minimum_connection(board, player, min_stones=6):
    visited = set()
    size = len(board)

    def dfs(r, c, count):
        if (r, c) in visited or count >= min_stones:
            return count
        visited.add((r, c))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size and board[rr][cc] == player:
                count = dfs(rr, cc, count + 1)
        return count

    for r in range(size):
        for c in range(size):
            if board[r][c] == player:
                if dfs(r, c, 1) >= min_stones:
                    return True
    return False


def can_win_next_move(game, action_coordinates, player):
    row, col = action_coordinates
    if not has_minimum_connection(game.board, player):
        return False

    game.board[row][col] = player
    has_won = game._evaluate_white(False)[0] if player == 1 else game._evaluate_black(False)[0]
    game.board[row][col] = 0 #undo move
    game.winner = 0 #undo winner from evaluate hex engine
    return has_won
