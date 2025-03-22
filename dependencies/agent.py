import random
import torch
import torch.nn.functional as F
import numpy as np
from .mcts import MCTS



class AgentLogic:
    def __init__(self, policy_net, device, q_threshold=0.5, mcts_simulations=2000, always_mcts=False, always_random=False):
        """
        Initialize the agent logic with a policy network, device, and a Q-value threshold.
        If the best Q-value for valid actions is below the threshold, the agent will fall back to MCTS.

        Args:
            policy_net: The DQN policy network.
            device: The device to run the model (CPU or GPU).
            q_threshold (float): Threshold for Q-value fallback to MCTS.
            mcts_simulations (int): Number of MCTS simulations per decision.
        """
        self.policy_net = policy_net
        self.device = device
        self.q_threshold = q_threshold
        self.mcts_simulations = mcts_simulations
        self.always_mcts = always_mcts
        self.always_random = always_random

    def pick_action(self, env, epsilon, logger, debug=False,mcts_fallback=True, evaluation=False):
        """
        Pick an action using an epsilon-greedy strategy with MCTS fallback.

        Args:
            env: The game environment.
            epsilon (float): Exploration probability.
            logger: Logger for debugging.
            debug (bool): Whether to output debug info.
            mcts_fallback (bool): If True, allow MCTS fallback; if False, use pure DQN.
            evaluation (bool): If True, MCTS will use DQN-based state evaluation.
        """
        mcts_taken = False
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return action, mcts_taken

        if self.always_random:
            action = random.choice(valid_actions)
            return action, mcts_taken

        if self.always_mcts:
            mcts_taken = True
            mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=None, 
                evaluation=False  # Use DQN-based evaluation if enabled
            )
            action, mcts_value = mcts_agent.select_action(env, env.current_player)
            return action, mcts_taken

        # Evaluate Q-values using the policy network.
        # Build the state tensor with the correct shape: (batch, channels, height, width)
        board_np = env.board  # Expected shape: (6,7)
        state_tensor = torch.tensor(board_np, dtype=torch.float32, device=self.device)
        if state_tensor.ndimension() == 2:
            # From (6,7) to (1,1,6,7)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        elif state_tensor.ndimension() == 3:
            # Assume it's (batch,6,7); add the channel dimension
            state_tensor = state_tensor.unsqueeze(1)

        # Forward pass to get Q-values; output shape should be (1,7)
        self.policy_net.eval()
        if evaluation:  # In evaluation mode, use MCTS with DQN-based evaluation.
            mcts_taken = True

            mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=self.policy_net, 
                evaluation=evaluation,
                q_threshold=self.q_threshold
            )
            action, mcts_value = mcts_agent.select_action(env, env.current_player)
            if debug:
                if mcts_value is not None:
                    logger.debug(f"Evaluation MCTS+DQN selected action {action} with MCTS value: {mcts_value:.3f}")
                    #print(f"Evaluation MCTS+DQN selected action {action} with MCTS value: {mcts_value:.3f}")
                else:
                    logger.debug(f"Evaluation MCTS+DQN selected action {action} with no MCTS value")
                    #print(f"Evaluation MCTS+DQN selected action {action} with no MCTS value")   
            return action, mcts_taken

 
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        self.policy_net.train()
        # Check if all q_values are nearly equal (or zero), then use a uniform distribution
        if np.allclose(q_values, q_values[0]):
            q_values = np.full_like(q_values, 1.0 / len(q_values))
        # Mask invalid actions: set Q-value for invalid actions to -inf
        masked_q = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        # Apply softmax to obtain normalized probabilities:
        exp_q = np.exp(masked_q - np.max(masked_q))  # subtract max for numerical stability
        softmax_q = exp_q / np.sum(exp_q)

        best_act = int(np.argmax(softmax_q))
        best_q_val = softmax_q[best_act]

        # If in inference mode without MCTS fallback, return the DQN best action.
        if not mcts_fallback:
            return action, mcts_taken

        # Branch 1:  Use MCTS with probability epsilon.
        if epsilon > self.q_threshold:
            self.q_threshold = epsilon

        # Branch 2: If best Q-value is below the threshold, fall back to MCTS.
        if best_q_val < self.q_threshold:
            if debug:
                logger.debug(f"Q-value {best_q_val:.3f} below threshold {self.q_threshold:.3f}, using MCTS fallback.")
                #print(f"Q-value {best_q_val:.3f} below threshold {self.q_threshold:.3f}, using MCTS fallback.")
            mcts_taken = True
            mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=self.policy_net, 
                evaluation=evaluation
            )
            action, mcts_value = mcts_agent.select_action(env, env.current_player)
            if debug:
                logger.debug(f"MCTS (low Q) selected action {action} with MCTS value: {mcts_value:.3f}")
                #print(f"MCTS (low Q) selected action {action} with MCTS value: {mcts_value:.3f}")
            return action, mcts_taken

        # Default: use the best action from DQN.
        if debug:
            logger.debug(f"Using DQN-selected action {best_act} with Q-value: {best_q_val:.3f}")
            #print(f"Using DQN-selected action {best_act} with Q-value: {best_q_val:.3f}")
        return best_act, mcts_taken

    def compute_reward(self, env, last_action, last_player):
        """
        Compute the reward for the last move.

        Args:
            env: The game environment.
            last_action: The action taken.
            last_player: The player (1 or 2).
            mcts_taken (bool): True if MCTS was used.
            epsilon (float): The exploration probability.
            low_q_value (bool): True if the Q-value was below threshold.

        Returns:
            tuple: (total_reward, win_status)
        """
        rs = RewardSystem()
        return rs.calculate_reward(env, last_action, last_player)

# ------------------ Reward Systems ------------------ #
class RewardSystem:
    def __init__(self, config=None):
        """
        Initialize the reward system with default or provided configuration.
        """
        self.config = config if config is not None else {
            "win": 100.0,
            "loss": -100.0,
            "draw": 0.0,
            "active_base": 1.0,
            "double_threat": 10.0,
            "block_four": 9.0,
            "cause_three": 5.0,
            "block_three": 7.0,
            "cause_two": 3.0,
            "block_two": 5.0,
            "center_bonus": 2.0,
        }   

    def calculate_reward(self, env, last_action, last_player):
        """
        Calculate the reward for the last move.
        """
        turn = env.turn - 1  # 0-indexed turn
        board = env.get_board()
        # Fix: set opponent to the other player (assuming players are 1 and 2)
        opponent = 3 - last_player

        winner = env.check_winner()
        if winner == last_player:
            #print("WIN")
            result_reward = self.config["win"]
            win_status = last_player
        elif winner == opponent:
            #print("LOSS")
            result_reward = self.config["loss"]
            win_status = opponent
        elif env.is_draw():
            #print("DRAW")
            result_reward = self.config["draw"]
            win_status = -1
        else:
            result_reward = 0.0
            win_status = 0

        fastest_win_possible = 8
        adjustment_factor = min(2, fastest_win_possible / (turn + 1))
        active_reward = self.get_active_reward(board, last_action, last_player)
        center_reward = self.get_center_bonus(board, last_action)
        # Use the corrected opponent here for passive penalty.
        passive_penalty = self.get_passive_penalty(board, opponent)
        total_reward =  (result_reward* adjustment_factor ) +active_reward + center_reward - (passive_penalty * adjustment_factor)

        return total_reward, win_status

    def get_active_reward(self, board, last_action, last_player):
        """
        Compute the reward for an active move based on various heuristics.
        You may choose to sum multiple contributions instead of returning immediately.
        """
        row_played = self.get_row_played(board, last_action)
        if row_played is None:
            return 0.0

        # Start with the base reward.
        reward = self.config["active_base"]

        # Use the column (last_action) for double threat check.
        if self.is_double_threat(board, last_action, last_player):
            return self.config["double_threat"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 4):
            return self.config["block_four"]
        if self.causes_n_in_a_row(board, row_played, last_action, last_player, 3):
            return self.config["cause_three"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 3):
            return self.config["block_three"]
        if self.causes_n_in_a_row(board, row_played, last_action, last_player, 2):
            return self.config["cause_two"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 2):
            return self.config["block_two"]
        return reward

    def get_passive_penalty(self, board, opponent):
        two_in_a_rows = self.count_n_in_a_row(board, opponent, 2)
        three_in_a_rows = self.count_n_in_a_row(board, opponent, 3)
        return (two_in_a_rows * 0.5) + (three_in_a_rows * 1.5)

    def get_center_bonus(self, board, col):
        center = board.shape[1] // 2
        return self.config["center_bonus"] if col == center else 0.0

    def get_row_played(self, board, col):
        rows = board.shape[0]
        for r in range(rows):
            if board[r, col] != 0:
                return r
        return None

    def is_double_threat(self, board, col_to_place, current_player):
        """
        Check if placing a piece in the given column creates a double threat.
        """
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
        for row in range(rows - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return True
        return False

    def find_valid_columns(self, board):
        return [col for col in range(board.shape[1]) if board[0, col] == 0]

    def check_if_winning_move(self, board, player):
        return self.four_in_a_row_exists(board, player)

    def four_in_a_row_exists(self, board, player):
        rows, cols = board.shape
        # Horizontal check.
        for r in range(rows):
            for c in range(cols - 3):
                if (board[r, c] == player and board[r, c+1] == player and
                    board[r, c+2] == player and board[r, c+3] == player):
                    return True
        # Vertical check.
        for r in range(rows - 3):
            for c in range(cols):
                if (board[r, c] == player and board[r+1, c] == player and
                    board[r+2, c] == player and board[r+3, c] == player):
                    return True
        # Diagonal (down-right) check.
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (board[r, c] == player and board[r+1, c+1] == player and
                    board[r+2, c+2] == player and board[r+3, c+3] == player):
                    return True
        # Diagonal (up-right) check.
        for r in range(3, rows):
            for c in range(cols - 3):
                if (board[r, c] == player and board[r-1, c+1] == player and
                    board[r-2, c+2] == player and board[r-3, c+3] == player):
                    return True
        return False

    def causes_n_in_a_row(self, board, row, col, player, n):
        return (
            self.check_line(board, row, col, player, n, "horizontal") or
            self.check_line(board, row, col, player, n, "vertical") or
            self.check_line(board, row, col, player, n, "diag1") or
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
        return False

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
