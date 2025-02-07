# agent.py
import random
import logging
import torch
import torch.nn.functional as F
from mcts import MCTS
from copy import deepcopy
import logging
import math
import numpy as np
class AgentLogic:
    def __init__(self, policy_net, device, q_threshold=0.5):
        self.policy_net = policy_net
        self.device = device
        self.q_threshold = q_threshold


    def pick_action(self, env, player, epsilon, episode=1, debug=False):
        """
        This function tries:
          1) with probability epsilon -> random
          2) q network
        """
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None

        # (1) Epsilon exploration
        if random.random() < epsilon:
            sims= 2000
            mcts_action = MCTS(num_simulations=sims, debug=True)
            action = mcts_action.select_action(env, env.current_player)
            if debug:
                logging.debug(f"MCTS Action SELECT={action}")
            return action

        # (2) Check Q Values
        self.policy_net.eval()

        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.policy_net.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

        # Restore original mode if needed
        self.policy_net.train()

        valid_actions = env.get_valid_actions()

        # Among valid actions, pick highest
        # Mask invalid actions by setting their Q-values to -inf
        masked_q = q_values.copy()
        masked_q[list(set(range(len(q_values))) - set(valid_actions))] = -np.inf

        best_act = np.argmax(masked_q)
        best_q_val = q_values[best_act]
        if debug:
            logging.debug(f"Q-vals = {masked_q}, best_act={best_act}, best_val={best_q_val:.3f}")
        return best_act

    def compute_reward(self, env, last_action, current_player):
        """
        External function that instantiates RewardSystem and calls its 
        'calculate_reward' method.
        
        Args:
            env: The current game environment.
            last_action: The action taken by the player.
            current_player: The player (1 or 2).
        
        Returns:
            tuple: (total_reward, win_status)
        """
        
        rs = RewardSystem()
        return rs.calculate_reward(env, last_action, current_player)

# ------------------ Reward Systems ------------------ #
class RewardSystem:
    def calculate_reward(self, env, last_action, current_player):
        turn = env.turn - 1  # Since the turn is already ended
        board = env.get_board()
        opponent = 3 - current_player

        winner = env.check_winner()
        if winner == current_player:
            result_reward = 100.0
            win_status = current_player
        elif winner == opponent:
            result_reward = -100.0
            win_status = current_player
        elif env.is_draw():
            result_reward = 10.0  # Lowered to prevent passive play
            win_status = -1
        else:
            result_reward = 0.0
            win_status = 0

        # Adjust scaling for turn-based reward
        fastest_win_possible = 8
        adjustment_factor = min(2, fastest_win_possible / (turn + 1))  # Smoother scaling

        # Active Rewards (Encourage Good Moves)
        active_reward = self.get_active_reward(board, last_action, current_player)

        # Passive Penalty (Discourage Allowing Opponent Advantage)
        passive_penalty = self.get_passive_penalty(board, opponent)

        # Total Reward Calculation
        raw_total = (result_reward * adjustment_factor) + active_reward - passive_penalty
        total_reward = raw_total

        return total_reward, win_status

    def get_active_reward(self, board, last_action, current_player):
        row_played = self.get_row_played(board, last_action)
        if row_played is None:
            return 0.0

        # More distinct reward scaling
        if self.is_double_threat(board, row_played, current_player):
            return 10.0
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 4):
            return 6.0
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 3):
            return 3.0
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 3):
            return 2.0
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 2):
            return 1.0
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 2):
            return 0.5
        return 0.2  # Minor incentive for placing a piece

    def get_passive_penalty(self, board, opponent):
        """Penalize allowing opponent to build connections."""
        two_in_a_rows = self.count_n_in_a_row(board, opponent, 2)
        three_in_a_rows = self.count_n_in_a_row(board, opponent, 3)
        return (two_in_a_rows * 0.1) + (three_in_a_rows * 0.8)  # Increased penalty for stronger threats


    def get_row_played(self, board, col):
        rows = board.shape[0]
        for r in range(rows):
            if board[r, col] != 0:
                return r
        return None

    def is_double_threat(self, board, col_to_place, current_player):
        temp_board = board.copy()
        if not self.place_piece(temp_board, col_to_place, current_player):
            return False
        winning_moves = 0
        for c in self.find_valid_columns(temp_board):
            next_board = temp_board.copy()
            if self.place_piece(next_board, c, current_player):
                if self.check_if_winning_move(next_board, current_player):
                    winning_moves += 1
            if winning_moves >= 2:
                return True
        return False

    def place_piece(self, board, col, player):
        if col < 0 or col >= board.shape[1] or board[0, col] != 0:
            return False
        rows = board.shape[0]
        for row in range(rows-1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return True
        return False

    def find_valid_columns(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if board[0, col] == 0:
                valid_cols.append(col)
        return valid_cols

    def check_if_winning_move(self, board, player):
        return self.four_in_a_row_exists(board, player)

    def four_in_a_row_exists(self, board, player):
        rows, cols = board.shape
        # (same 4-in-a-row logic)
        for r in range(rows):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r, c+1] == board[r, c+2] == board[r, c+3]):
                    return True
        for r in range(rows - 3):
            for c in range(cols):
                if (board[r, c] == player == board[r+1, c] == board[r+2, c] == board[r+3, c]):
                    return True
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]):
                    return True
        for r in range(rows - 3):
            for c in range(3, cols):
                if (board[r, c] == player == board[r+1, c-1] == board[r+2, c-2] == board[r+3, c-3]):
                    return True
        return False

    def causes_n_in_a_row(self, board, row, col, player, n):
        return (
            self.check_line(board, row, col, player, n, "horizontal") or
            self.check_line(board, row, col, player, n, "vertical")   or
            self.check_line(board, row, col, player, n, "diag1")      or
            self.check_line(board, row, col, player, n, "diag2")
        )

    def blocks_opponent_n_in_a_row(self, board, row, col, current_player, n):
        opponent = 3 - current_player
        original_value = board[row, col]
        board[row, col] = 0
        caused = self.causes_n_in_a_row(board, row, col, opponent, n)
        board[row, col] = original_value
        return caused

    def check_line(self, board, row, col, player, n, direction):
        rows, cols = board.shape
        def count_consecutive(r_step, c_step):
            count = 1
            rr, cc = row + r_step, col + c_step
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                count += 1
                rr += r_step
                cc += c_step
            rr, cc = row - r_step, col - c_step
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                count += 1
                rr -= r_step
                cc -= c_step
            return count

        if direction == "horizontal":
            return count_consecutive(0, 1) >= n
        elif direction == "vertical":
            return count_consecutive(1, 0) >= n
        elif direction == "diag1":
            return count_consecutive(1, 1) >= n
        elif direction == "diag2":
            return count_consecutive(1, -1) >= n

    def count_n_in_a_row(self, board, player, n):
        rows, cols = board.shape
        visited_segments = set()
        total_count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def in_bounds(r, c):
            return 0 <= r < rows and 0 <= c < cols

        for r in range(rows):
            for c in range(cols):
                if board[r, c] == player:
                    for dr, dc in directions:
                        cells_in_line = []
                        rr, cc = r, c
                        while in_bounds(rr, cc) and board[rr, cc] == player:
                            cells_in_line.append((rr, cc))
                            rr += dr
                            cc += dc
                        if len(cells_in_line) >= n:
                            for start_idx in range(len(cells_in_line) - n + 1):
                                segment = tuple(cells_in_line[start_idx:start_idx + n])
                                if segment not in visited_segments:
                                    visited_segments.add(segment)
                                    total_count += 1
        return total_count
