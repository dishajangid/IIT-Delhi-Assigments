import time
import math
import random
import numpy as np
from helper import *

class AIPlayer:
    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer Agent.

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game.
        `timer`: Timer object that can be used to fetch the remaining time for any player.
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = f'Player {player_number}: ai'
        self.timer = timer
        self.move_count = 0  # Keep track of how many moves the AI has made
        self.alternating_corners = []  # To store the alternating corner sequence

    def get_valid_alternating_corners(self, state: np.array) -> list:
        """
        Get a list of available alternating corner moves.
        """
        if not self.alternating_corners:
            # Define the alternating corner sequence
            dim = state.shape[0]
            corners = get_all_corners(dim)
            self.alternating_corners = [corners[i % len(corners)] for i in range(0, len(corners), 2)]  # Alternate corners

        valid_corner_moves = [corner for corner in self.alternating_corners if state[corner] == 0]
        return valid_corner_moves

    def get_move(self, state: np.array) -> tuple[int, int]:
        """
        Use Minimax to get the best move for the AI.
        First three moves prioritize alternating corners.
        """
        if self.move_count < 3:
            # Get valid alternating corner moves
            corner_moves = self.get_valid_alternating_corners(state)
            if corner_moves:
                self.move_count += 1
                return random.choice(corner_moves)  # Choose a random available alternating corner move

        # After 3 moves, revert to regular Minimax strategy
        depth = 3
        _, best_move = self.minimax(state, depth, True, -math.inf, math.inf)

        if best_move is not None:
            self.move_count += 1
            return best_move
        else:
            raise Exception("No valid move found")

    def evaluate_board(self, state: np.array) -> int:
        """
        Comprehensive evaluation function considering:
        - Control of edges and corners
        - Proximity to winning structures (forks, bridges, rings)
        - Blocking opponent's proximity to winning structures
        - Prioritize forming AI's own winning structures
        """
        score = 0
        ai_count = np.sum(state == self.player_number)
        opponent_count = np.sum(state == (3 - self.player_number))

        # Basic scoring: AI cells +1, opponent cells -1
        score += ai_count - opponent_count

        # Prioritize ring formations for AI and block opponent's ring
        if self.detect_ring(state, self.player_number):
            score += 300  # Encourage forming a ring
        if self.detect_ring(state, 3 - self.player_number):
            score -= 300  # Block opponent's ring

        # Detect AI's and opponent's potential bridges and forks
        if self.detect_bridge(state, self.player_number):
            score += 100  # Encourage forming a bridge
        if self.detect_fork(state, self.player_number):
            score += 100  # Encourage forming a fork

        # Blocking opponent’s forks and bridges with higher priority on forks
        if self.detect_bridge(state, 3 - self.player_number):
            score -= 150  # Block opponent's bridge
        if self.detect_fork(state, 3 - self.player_number):
            score -= 250  # Increase penalty to block opponent's fork

        # Immediate threat detection for opponent's forks and bridges
        if self.opponent_near_fork(state, 3 - self.player_number):
            score -= 400  # Further increase penalty if opponent is about to form a fork
        if self.opponent_near_bridge(state, 3 - self.player_number):
            score -= 200  # Moderate penalty if opponent is about to form a bridge

        return score

    def detect_bridge(self, state: np.array, player: int) -> bool:
        """
        Detect if the player is close to forming a bridge (connecting two opposite corners).
        Returns True if a bridge is close, False otherwise.
        """
        dim = state.shape[0]
        corners = get_all_corners(dim)
        visited = set()

        def dfs(cell, parent):
            visited.add(cell)
            neighbors = self.get_neighbors(cell, dim)
            for neighbor in neighbors:
                if state[neighbor] == player:
                    if neighbor not in visited:
                        if dfs(neighbor, cell):
                            return True
                    elif neighbor != parent:
                        return True
            return False

        # If two corners are connected by the player's pieces, return True
        for corner in corners:
            if state[corner] == player:
                if dfs(corner, None):
                    return True
        return False

    def detect_fork(self, state: np.array, player: int) -> bool:
        """
        Detect if the player is close to forming a fork (connecting three edges).
        Returns True if a fork is close, False otherwise.
        """
        dim = state.shape[0]
        edges = get_all_edges(dim)
        controlled_edges = set()

        # Check which edges the player is controlling
        for edge in edges:
            for cell in edge:
                if state[cell] == player:
                    controlled_edges.add(tuple(edge))

        # If the player controls two or more edges and can form a fork, return True
        if len(controlled_edges) >= 2:
            for edge in edges:
                if tuple(edge) not in controlled_edges:
                    for cell in edge:
                        if state[cell] == 0:  # If there's an empty cell
                            return True
        return False

    def detect_ring(self, state: np.array, player: int) -> bool:
        """
        Detect if the player is close to forming a ring (a closed loop of pieces).
        Returns True if a ring is close, False otherwise.
        """
        dim = state.shape[0]
        visited = set()

        def dfs(cell, parent):
            visited.add(cell)
            neighbors = self.get_neighbors(cell, dim)
            for neighbor in neighbors:
                if state[neighbor] == player:
                    if neighbor not in visited:
                        if dfs(neighbor, cell):
                            return True
                    elif neighbor != parent:
                        return True
            return False

        # Check for a cycle (ring) for each unvisited cell of the player
        for i in range(dim):
            for j in range(dim):
                if state[i, j] == player and (i, j) not in visited:
                    if dfs((i, j), None):
                        return True
        return False

    def opponent_near_fork(self, state: np.array, opponent: int) -> bool:
        """
        Detect if the opponent is one move away from forming a fork.
        Returns True if an immediate threat exists, False otherwise.
        """
        dim = state.shape[0]
        edges = get_all_edges(dim)
        controlled_edges = set()

        # Check which edges the opponent is controlling
        for edge in edges:
            for cell in edge:
                if state[cell] == opponent:
                    controlled_edges.add(tuple(edge))

        # If opponent controls two or more edges and has an empty cell to complete a fork
        if len(controlled_edges) >= 2:
            for edge in edges:
                if tuple(edge) not in controlled_edges:
                    for cell in edge:
                        if state[cell] == 0:  # If there's an empty cell
                            return True
        return False

    def opponent_near_bridge(self, state: np.array, opponent: int) -> bool:
        """
        Detect if the opponent is one move away from forming a bridge.
        Returns True if an immediate threat exists, False otherwise.
        """
        dim = state.shape[0]
        corners = get_all_corners(dim)
        visited = set()

        def dfs(cell, parent):
            visited.add(cell)
            neighbors = self.get_neighbors(cell, dim)
            for neighbor in neighbors:
                if state[neighbor] == opponent:
                    if neighbor not in visited:
                        if dfs(neighbor, cell):
                            return True
                    elif neighbor != parent:
                        return True
            return False

        for corner in corners:
            if state[corner] == opponent:
                if dfs(corner, None):
                    return True

        return False

    def detect_ring_threat(self, state: np.array, player: int) -> bool:
        """
        Detect if the opponent is close to forming a ring.
        Uses cycle detection in an undirected graph where each node is a cell and edges connect neighbors.

        Returns True if a ring is close to being formed, False otherwise.
        """
        dim = state.shape[0]
        visited = set()

        def dfs(cell, parent):
            visited.add(cell)
            neighbors = self.get_neighbors(cell, dim)

            for neighbor in neighbors:
                if state[neighbor] == player:
                    if neighbor not in visited:
                        if dfs(neighbor, cell):
                            return True
                    elif neighbor != parent:
                        return True
            return False

        # Check for a cycle for each unvisited cell of the player
        for i in range(dim):
            for j in range(dim):
                if state[i, j] == player and (i, j) not in visited:
                    if dfs((i, j), None):
                        return True

        return False

    def detect_fork_threat(self, state: np.array, player: int) -> bool:
        """
        Detect if the opponent is close to forming a fork (connecting three edges).

        Returns True if the opponent is close to forming a fork, False otherwise.
        """
        dim = state.shape[0]
        edges = get_all_edges(dim)

        # Check which edges the opponent is controlling
        controlled_edges = set()
        for edge in edges:
            for cell in edge:
                if state[cell] == player:
                    controlled_edges.add(tuple(edge))

        # If the opponent controls two or more edges and can form a fork, return True
        if len(controlled_edges) >= 2:
            for edge in edges:
                if tuple(edge) not in controlled_edges:
                    # Check if there's a way to connect the opponent's current edges to this third edge
                    for cell in edge:
                        if state[cell] == 0:  # If there's an empty cell
                            return True

        return False

    def get_neighbors(self, position: tuple[int, int], dim: int) -> list:
        """
        Get the valid neighbors of a cell in the hexagonal grid.

        `position`: A tuple (i, j) representing the current cell's coordinates.
        `dim`: The dimensions of the grid.

        Returns a list of neighbor coordinates.
        """
        i, j = position
        neighbors = [
            (i-1, j), (i+1, j), (i, j-1), (i, j+1),
            (i-1, j+1), (i+1, j-1)  # Diagonals in a hex grid
        ]
        # Filter out invalid neighbors that are out of bounds
        return [(x, y) for x, y in neighbors if 0 <= x < dim and 0 <= y < dim]

    def is_terminal(self, state: np.array) -> bool:
        """
        Check whether the game has reached a terminal state.
        The game ends if:
        - A player forms a winning structure (Fork, Bridge, or Ring).
        - No valid moves are left (draw).
        """
        if len(get_valid_actions(state)) == 0:
            return True

        # Check if either player has won
        for player_num in [1, 2]:
            for move in np.argwhere(state == player_num):
                if check_win(state, tuple(move), player_num)[0]:
                    return True
        return False

    def simulate_move(self, state: np.array, action: tuple[int, int], player_number: int) -> np.array:
        """
        Simulate a move by placing a piece on the board at the given action (position).

        `state`: Current board state as a numpy array.
        `action`: A tuple (i, j) indicating the row and column where the piece should be placed.
        `player_number`: The player making the move (either 1 or 2).

        Returns a new board state with the move applied.
        """
        new_state = np.copy(state)
        new_state[action] = player_number
        return new_state

    def minimax(self, state: np.array, depth: int, maximizing_player: bool, alpha: float, beta: float):
        """
        Minimax algorithm with alpha-beta pruning to find the best move.

        `state`: Current board state.
        `depth`: Depth to limit the search.
        `maximizing_player`: True if it's the AI's turn, False for opponent.
        `alpha`: Best value for the maximizer so far.
        `beta`: Best value for the minimizer so far.

        Returns a tuple (evaluation, best_move).
        """
        if depth == 0 or self.is_terminal(state):
            return self.evaluate_board(state), None

        valid_actions = get_valid_actions(state)

        if maximizing_player:
            max_eval = -math.inf
            best_move = None
            for action in valid_actions:
                new_state = self.simulate_move(state, action, self.player_number)
                eval, _ = self.minimax(new_state, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            for action in valid_actions:
                new_state = self.simulate_move(state, action, 3 - self.player_number)
                eval, _ = self.minimax(new_state, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
