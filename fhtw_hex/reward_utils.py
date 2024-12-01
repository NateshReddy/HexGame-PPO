import numpy as np

class UnionFind:
    """
    A Union-Find (Disjoint-Set) data structure that supports efficient operations
    to find the root of a set and unite two sets. This implementation includes
    path compression and rank optimization to keep the tree structures shallow.

    Attributes:
        parent (List[int]): Parent list where parent[i] is the parent of element i.
                            If parent[i] == i, then i is the root of its set.
        rank (List[int]): Rank list to track the depth of the tree rooted at each element.
    """

    def __init__(self, n):
        """
        Initializes the Union-Find data structure with `n` elements.

        Each element is initially its own parent, representing `n` individual sets.
        The rank of all elements is initialized to 0.

        Args:
            n (int): The number of elements in the set, indexed from 0 to n-1.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """
        Finds the root of the set containing the element `x` with path compression.

        Path compression ensures that all elements on the path from `x` to the root
        point directly to the root, optimizing future operations.

        Args:
            x (int): The element whose set root is to be found.

        Returns:
            int: The root of the set containing `x`.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Unites the sets containing elements `x` and `y` using rank optimization.

        The root of one set becomes the parent of the root of the other set based
        on the rank of the roots. This helps keep the tree structures shallow.

        Args:
            x (int): An element in the first set.
            y (int): An element in the second set.

        Returns:
            None
        """
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
                
def will_opponent_win_next(game, player, min_stones=6):
    opponent = -player
    size = len(game.board)

    # Check if the opponent has enough stones placed
    opponent_stones = np.sum(game.board == opponent)  # Use np.sum instead of row.count
    if opponent_stones < min_stones:
        return None

    for r in range(size):
        for c in range(size):
            if game.board[r][c] == 0:  # Empty cell
                game.board[r][c] = opponent
                if opponent == 1:
                    has_won, _ = game._evaluate_white(False)
                else:
                    has_won, _ = game._evaluate_black(False)
                game.board[r][c] = 0
                game.winner = 0  # Reset winner due to evaluation methods
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
        if (0 <= r1 < board.shape[0] and 0 <= c1 < board.shape[1]) and (
                0 <= r2 < board.shape[0] and 0 <= c2 < board.shape[1]):
            # Check if both surrounding positions contain opponent's stones
            if board[r1, c1] == opponent and board[r2, c2] == opponent:
                # Ensure the middle position (action_coordinates) is empty
                return True
    return False


def is_adjecent_move(board, action_coordinates, player):
    """
    Checks if a move connects to another of the player's stones.
    """
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
    """
    Determines if the given move can lead to a win for the player.

    Args:
        game: The game instance (OurHexGame or similar).
        action_coordinates: A tuple (row, col) representing the move coordinates.
        player: The current player (1 for player_1, 2 for player_2).

    Returns:
        bool: True if the move results in a win for the player, False otherwise.
    """
    row, col = action_coordinates
    if game.board[row, col] != 0:
        # The cell is already occupied, so it's not a valid move
        return False

    # Simulate the move
    game.board[row, col] = player

    # Check if the player wins after this move
    has_won = game.check_winner(player)

    # Undo the move
    game.board[row, col] = 0
    game.uf = UnionFind(game.board_size * game.board_size + 4)  # Reset union-find structure
    game.winner = 0  # Reset winner

    return has_won

def is_connecting_move(board, action_coordinates, player):
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
        if (0 <= r1 < board.shape[0] and 0 <= c1 < board.shape[1]) and (
                0 <= r2 < board.shape[0] and 0 <= c2 < board.shape[1]):
            # Check if both surrounding positions contain opponent's stones
            if board[r1, c1] == player and board[r2, c2] == player:
                # Ensure the middle position (action_coordinates) is empty
                return True
    return False


def compute_rewards(game, action_coordinates, player):
    """
    Compute the reward for a given move based on the game's state and the action taken.

    Args:
        game: The game instance (OurHexGame or similar).
        action_coordinates: A tuple (row, col) representing the action taken.
        player: The current player making the move (1 or -1).

    Returns:
        float: The computed reward for the move.
    """
    # Check if the move results in a win
    if can_win_next_move(game, action_coordinates, player):
        import ipdb; ipdb.set_trace()
        return 60  # Reward for winning the game

    # Check if the move results in a loss for the player
    if player == 1:
        opponent = 2
    else:
        opponent = 1
    if can_win_next_move(game, action_coordinates, opponent):
        import ipdb; ipdb.set_trace
        return -60  # Penalty for losing the game

    # If it was not a winning or losing move
    return -1