import random
import torch
import torch.nn.functional as F
import numpy as np
from .mcts import MCTS

class AgentLogic:
    def __init__(self, policy_net, device, q_threshold=0.9, mcts_simulations=2000, always_mcts=False, always_random=False):
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
        self.hybrid_threshold=0.7   #hybrid_threshold
        self.temperature = 0.1
        self.mcts_simulations = mcts_simulations
        self.always_mcts = always_mcts
        self.always_random = always_random
        self.hybrid_value_threshold =1.1 # weight on mcts_value  to compare with hybrid

    def pick_action(self, env, epsilon, logger, debug=False,mcts_fallback=True, hybrid=False):
        """
        Pick an action using an epsilon-greedy strategy with MCTS fallback.

        Args:
            env: The game environment.
            epsilon (float): Exploration probability.
            logger: Logger for debugging.
            debug (bool): Whether to output debug info.
            mcts_fallback (bool): If True, allow MCTS fallback; if False, use pure DQN.
            hybrid (bool): If True, MCTS will use DQN-based state hybrid.
        Return:
         tuple:
            q_action(int): dqn based action
            mcts_action(int): mcts action either from hybrid or mcts 
            mcts_value(float): mcts win confidence
            model_used: (String): mcts, dqn, hybrid, random
        """
        q_action = mcts_action = mcts_value = None
        model_used = None

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return -1, -1, -1,model_used

        if self.always_random:  # We only use this for Hybrid Purpose Only
            random_action = random.choice(valid_actions)
            model_used="random"
            model_used,q_action, None, None, None, None, None,random_action

        if self.always_mcts:# We only use this for Hybrid Purpose Only
            model_used = "mcts"
            mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=None, 
                hybrid=False  # Use DQN-based hybrid if enabled
            )
            mcts_action, mcts_value = mcts_agent.select_action(env, env.current_player)
            return model_used,None, mcts_action, None, None, None, None,None
                
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

        # Adjust senstivity og softmax with probability epsilon.

        if epsilon> self.q_threshold:
              self.q_threshold = epsilon

        if epsilon> self.temperature:
              self.temperature = epsilon

        # Compute raw Q-values with gradient tracking.
        q_values = self.policy_net(state_tensor).flatten()  # Assuming state_tensor is for a single state

        # Check if all q_values are nearly equal; if so, use a uniform distribution.
        if torch.allclose(q_values, q_values[0].expand_as(q_values)):
            q_values = torch.full_like(q_values, 1.0 / q_values.numel())

        # Mask invalid actions: set invalid actions to -inf so they receive zero probability.
        masked_q = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            masked_q[a] = q_values[a]

        # Adjust temperature dynamically (assuming self.temperature is a float)
        dynamic_temp = min(epsilon * 2, 1)
        if dynamic_temp > self.temperature:
            self.temperature = dynamic_temp

        # Compute softmax with temperature scaling in a differentiable manner.
        softmax_q = torch.nn.functional.softmax(masked_q / self.temperature, dim=0)

        # Option 1: Stochastically sample an action from the softmax distribution.
        sampled_action_tensor = torch.multinomial(softmax_q, num_samples=1)
        q_action = sampled_action_tensor.item()

        # Option 2 (for debugging): Also compute the best (argmax) action.
        best_act = torch.argmax(softmax_q)
        best_q_val = softmax_q[best_act].detach()  # Detach for logging or serialization





        # Make the q-threshold dynamic till the lowest point
        # If in inference mode without MCTS fallback, return the DQN best action.
        if not mcts_fallback:
            model_used="dqn"
            return model_used,q_action, None, None, None, None, None,None

        hybrid=True
        mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=self.policy_net, 
                hybrid=hybrid,# In hybrid mode, use MCTS with DQN-based hybrid.
                q_threshold=self.q_threshold

            )
        hybrid_action, hybrid_value = mcts_agent.select_action(env, env.current_player)

        hybrid=False
        mcts_agent = MCTS(
                logger=logger, 
                num_simulations=self.mcts_simulations, 
                debug=debug, 
                dqn_model=self.policy_net, 
                hybrid=hybrid,# In hybrid mode, use MCTS with DQN-based hybrid.
                q_threshold=self.q_threshold

            )
        mcts_action, mcts_value = mcts_agent.select_action(env, env.current_player)



        if hybrid_value> mcts_value*self.hybrid_value_threshold : # 1.1* hybrid >mcts_value so hybrid value has to be little more tha 10% than mcts_value
            hybrid=True
            model_used = "hybrid"
        else:
            hybrid=False
            model_used= "mcts"
        if best_q_val > self.q_threshold:
            model_used="dqn"
        
        if debug:
            logger.debug(f"Model used: {model_used}, Q Action: {best_act.item()}, MCTS Action: {mcts_action}, Hybrid Action: {hybrid_action}, Best Q Value: {best_q_val.item()}, MCTS Value: {mcts_value}, Hybrid Value: {hybrid_value}")    
            print(f"Model used: {model_used}, Q Action: {best_act.item()}, MCTS Action: {mcts_action}, Hybrid Action: {hybrid_action}, Best Q Value: {best_q_val.item()}, MCTS Value: {mcts_value}, Hybrid Value: {hybrid_value}")        
        return model_used, q_action, mcts_action,hybrid_action, best_q_val, mcts_value,hybrid_value,None
        # Hybrid used(hybrid), dqn_used=> model_used as string: mcts, dqn, hybrid
    def compute_reward(self, env, last_action, last_player):
        """
        Compute the reward for the last move.

        Args:
            env: The game environment.
            last_action: The action taken.
            last_player: The player (1 or 2).
            mcts_used (bool): True if MCTS was used.
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
            "draw": 1.0,
            "active_base": 0.1,
            "ignore_four_in_row": -5.0,
            "double_threat": 4.5,
            "block_four": 4.0,
            "cause_three": 3.5,
            "block_three": 3.0,
            "cause_two": 2.5,
            "block_two": 2.0,
            "center_bonus": 0.5,
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
        reward = self.config["active_base"]
        row_played = self.get_row_played(board, last_action)
        if row_played is None:
            return 0.0

        if self.is_ignore_four_in_row(board, last_action, last_player):
            reward += self.config["ignore_four_in_row"]
        if self.is_double_threat(board, last_action, last_player):
            reward += self.config["double_threat"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 4):
            reward += self.config["block_four"]
        if self.causes_n_in_a_row(board, row_played, last_action, last_player, 3):
            reward += self.config["cause_three"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 3):
            reward += self.config["block_three"]
        if self.causes_n_in_a_row(board, row_played, last_action, last_player, 2):
            reward += self.config["cause_two"]
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, last_player, 2):
            reward += self.config["block_two"]

        return reward


    def get_passive_penalty(self, board, opponent):
        two_in_a_rows = self.count_n_in_a_row(board, opponent, 2)
        three_in_a_rows = self.count_n_in_a_row(board, opponent, 3)
        return (two_in_a_rows * 1) + (three_in_a_rows * 2)

    def get_center_bonus(self, board, col):
        center = board.shape[1] // 2
        return self.config["center_bonus"] if col == center else 0.0

    def get_row_played(self, board, col):
        rows = board.shape[0]
        # Iterate from the bottom to the top to get the most recently placed piece.
        for r in range(rows - 1, -1, -1):
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



    def is_ignore_four_in_row(self, board, col_to_takeback, current_player):
        """
        First, remove (take back) the last piece placed by current_player in col_to_takeback.
        Then, for each column (assumed to be 7 total, or dynamically using board.shape[1]),
        simulate placing a piece for current_player and check if it would be a winning move.
        Returns True if any such winning move is found, otherwise False.
        """
        temp_board = board.copy()
        
        # Remove the last piece placed in the specified column.
        new_board = self.takeback_piece(temp_board, col_to_takeback, current_player)
        if new_board is False:
            return False  # Could not take back the piece (e.g. wrong column or piece not found)
        
        # Brute force: Try each column as a potential move.
        for c in range(new_board.shape[1]):  # or simply use range(7) if board is 7 columns wide
            candidate_board = new_board.copy()
            if self.place_piece(candidate_board, c, current_player):
                if self.check_if_winning_move(candidate_board, current_player):
                    return True
        return False


    def takeback_piece(self, board, col, player):
        """
        Remove the most recently placed piece (i.e. the bottom-most nonzero piece)
        from the given column if that piece belongs to the specified player.
        """
        temp_board = board.copy()
        if col < 0 or col >= temp_board.shape[1]:
            return False

        # Iterate from bottom to top to remove the most recent piece.
        for row in range(temp_board.shape[0] - 1, -1, -1):
            if temp_board[row, col] != 0:
                if temp_board[row, col] == player:
                    temp_board[row, col] = 0  # Remove the piece.
                    return temp_board
                else:
                    return False  # The piece in this column does not belong to player.
        return False  # No piece found in the column.

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
