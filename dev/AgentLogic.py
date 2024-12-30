import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import random
from copy import deepcopy
# Project-Specific Imports
from connect4 import Connect4  # For the Connect4 game class
from DQN import DQN  # For Deep Q-Networks
from concurrent.futures import ThreadPoolExecutor
import threading
from config import EPSILON
import logging
from logger_utils import setup_logger
class AgentLogic:
    def __init__(self,policy_net):
        """
        Initialize the AgentLogic class with a policy network and device.

        :param policy_net: The neural network model for predicting Q-values.
        """
        self.policy_net=policy_net

    def get_win_move(self, env, player,debug=1):
        """
        Check for a winning move for the given player before any move is selected.
        Returns the column index of the move or None if no such move exists.
        """
        for col in env.get_valid_actions():
            temp_env = env.copy()
            temp_env.make_move(col, warning=0)  # Simulate the move
            #logging.debug(temp_env.get_board())
            if temp_env.check_winner() == player:
                if debug==1:
                    logging.debug(f"Player {player} can win by placing in column {col}.")
                return col
        if debug==1:
            logging.debug(f"No winning move found for Player {player}.")
        return None

    def get_block_move(self, env, player,debug=1):
        """
        Check for a blocking move to prevent the opponent from winning.
        Returns the column index of the move or None if no such move exists.
        """
        opponent = 3 - player  # Determine the opponent
        valid_actions = env.get_valid_actions()
        if debug==1:
            logging.debug(f"Checking blocking moves for Player {player}. Valid actions: {valid_actions}")

        for col in valid_actions:
            temp_env = env.copy()  # Copy the environment
            temp_env.changeTurn() # change to opponent's move
            temp_env.make_move(col, warning=0)  # Simulate opponent's move
            #logging.debug(temp_env.get_board())

            # Check if the opponent would win in this column
            if temp_env.check_winner() == opponent:
                if debug==1:
                    logging.debug(f"Player {player} can block opponent's win in column {col}.")
                return col  # Block the opponent's winning move
        if debug==1:
            logging.debug(f"No blocking move found for Player {player}.")
        return None


    def logic_based_action(self, env, current_player,debug=1):
        """
        Use logic to decide the move (winning or blocking).
        If no logical move exists, return None.
        """

        # Check for a winning move
        win_move = self.get_win_move(env, current_player,debug)
        if win_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a winning move in column {win_move}.")
            return win_move

        # Check for a blocking move
        block_move = self.get_block_move(env, current_player,debug)
        if block_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a blocking move in column {block_move}.")
            return block_move

        # No logical move found
        if debug==1:
            logging.debug(f"Player {current_player} found no logical move.")
        return None




    # Updated monte_carlo_tree_search with parallelism
    def monte_carlo_tree_search(self, env, num_simulations=1000):
        """
        Monte Carlo Tree Search with parallel simulations.
        """
        class MCTSNode:
            def __init__(self, state, parent=None):
                self.state = state.copy()
                self.parent = parent
                self.children = []
                self.visits = 0
                self.wins = 0
                self.lock = threading.Lock()  # Lock for thread-safe updates

            def ucb_score(self, exploration_constant=1.414):
                if self.visits == 0:
                    return float('inf')  # Encourage exploration of unvisited nodes
                if self.parent.visits == 0:
                    return 0  # If the parent has no visits, return 0
                
                win_rate = self.wins / self.visits
                exploration_term = exploration_constant * (
                    np.sqrt(max(0, np.log(max(1, self.parent.visits))) / self.visits)
                )
                return win_rate + exploration_term


            def update(self, win_result):
                """
                Thread-safe update for visits and wins.
                """
                with self.lock:
                    self.visits += 1
                    if win_result == 1:
                        self.wins += 1

        root = MCTSNode(env)

        def simulate(node):
            """
            Perform a single MCTS simulation.
            """
            current_node = node

            # Selection: Traverse tree using UCB
            while current_node.children:
                current_node = max(current_node.children, key=lambda n: n.ucb_score())

            # Expansion: Create child nodes if not terminal
            if not current_node.children and not current_node.state.check_winner():
                valid_actions = current_node.state.get_valid_actions()
                for move in valid_actions:
                    temp_env = current_node.state.copy()
                    logicBased =self.logic_based_action(temp_env,temp_env.current_player,0)#check 3 in rows
                    if logicBased is not None:
                        move= logicBased 
                    temp_env.make_move(move, 0)
                    child_node = MCTSNode(temp_env, parent=current_node)
                    current_node.children.append(child_node)

            # Simulation: Randomly play out the game
            current_state = current_node.state.copy()
            while not current_state.check_winner() and not current_state.is_draw():
                valid_actions = current_state.get_valid_actions()
                if not valid_actions:
                    break
                logicBased =self.logic_based_action(current_state,current_state.current_player,0)#check 3 in rows
                if logicBased is not None:
                    move= logicBased 
                move = random.choice(valid_actions)
                current_state.make_move(move, 0)

            # Backpropagation: Update visits and wins
            winner = current_state.check_winner()
            while current_node is not None:
                win_result = 1 if winner == 2 else -1 if winner == 1 else 0
                current_node.update(win_result)
                current_node = current_node.parent

        # Use ThreadPoolExecutor to run simulations in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(simulate, root) for _ in range(num_simulations)]
            for future in futures:
                future.result()  # Wait for all simulations to complete

        # Select the best move based on visits
        if not root.children:
            logging.warning("MCTS failed to generate children nodes. Falling back to random valid action.")
            valid_actions = env.get_valid_actions()
            return random.choice(valid_actions) if valid_actions else None

        best_child = max(root.children, key=lambda n: n.visits)
        return env.get_valid_actions()[root.children.index(best_child)]

    def normalize_q_values(self,q_values):
        """
        Normalize Q-values to a probability distribution using softmax.
        
        Args:
            q_values (torch.Tensor or np.ndarray): Input Q-values.
            
        Returns:
            torch.Tensor: Q-values normalized to a probability distribution.
        """
        if isinstance(q_values, np.ndarray):  # Convert NumPy array to PyTorch tensor
            q_values = torch.tensor(q_values, dtype=torch.float32)

        # Apply softmax to normalize Q-values into probabilities
        q_values_softmax = torch.softmax(q_values, dim=0)
        return q_values_softmax


    def combined_action(self, env,current_episode):
        """
        Decide the action for the AI (Player 2) using logical rules, MCTS, and DQN.
        """
        current_player = env.current_player
        device = self.policy_net.device  # Access the device from the policy_net
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()
        q_values =self.normalize_q_values(q_values)
        valid_actions = env.get_valid_actions()
        valid_q_values = {action: q_values[action] for action in valid_actions}
        
        formatted_q_values = {action: f"{q_value:.3f}" for action, q_value in valid_q_values.items()}
        logging.debug(f"Q-values: {formatted_q_values}")
        # Select the action with the highest Q-value
        action = max(valid_q_values, key=lambda a: valid_q_values[a])
        max_q_value = valid_q_values[action]
        
        if max_q_value < 0.5 or EPSILON > 0.5:
            # Use logic-based or MCTS if Q-values are too low or begging of stage
            if current_episode >10000:# Set the cap for the number of simulation
                set_simulations=10000
            else:
                set_simulations=current_episode
            mcts = self.monte_carlo_tree_search(env, num_simulations= set_simulations)
            if mcts is not None:
                logging.debug(f"Player{current_player}: Using MCTS for action: {mcts}")
                return mcts
        logging.debug(f"Player{current_player}:selected: Column {action}, Q-value: {max_q_value:.3f}")
        return action

    def calculate_reward(self, env, last_action, current_player):
        """
        Calculate the reward for the given player based on the game state and potential outcomes.
        
        Args:
            env: The current game environment.
            action: The action taken by the player.
            current_player: The player (1 or 2) for whom the reward is calculated.
            
        Returns:
            tuple: (reward, win_status), where:
                - reward (float): The calculated reward.
                - win_status (int): 1 if current_player wins, -1 if opponent wins, 0 otherwise.
        """
        opponent = 3 - current_player  # Determine the opponent's player ID

        if env.check_winner() == current_player:
            logging.debug(f"Player {current_player} wins!")
            return 1.0, current_player  # Reward for winning and win status
        elif env.check_winner() == opponent:
            logging.debug(f"Player {current_player} loses!")
            return -1.0, opponent  # Penalty for losing and loss status
        elif env.is_draw():
            logging.debug("Game is a draw!")
            return 0.0, 0  # Neutral reward and no win status
        
        # Additional rewards for strategic moves
        logging.debug("Checking if there is any 3 in rows")
        if self.detect_double_three_in_a_row(env, current_player):
            logging.debug(f"Action {last_action} made two '3 in a row' patterns for Player {current_player}.")
            return 5.0, 0  # High reward, no immediate win
        elif self.is_WinBlock_move(env,last_action,current_player):
            logging.debug(f"Action {last_action} is a winning or blocking move for Player {opponent}.")
            return 0.5, 0  # Small reward, no immediate win

        # Small penalty for non-advantageous moves
        return -0.01, 0  # Neutral move, no immediate win


    def detect_double_three_in_a_row(self, env, current_player):
        """
        Check if there are two separate "3 in a row" patterns on the board for the given player,
        where each pattern has one empty slot that can be filled to create a "4 in a row."

        Args:
            env (Connect4): The current game environment.
            current_player (int): The player to check for.

        Returns:
            bool: True if there are two distinct "3 in a row" patterns, False otherwise.
        """
        board = env.get_board()
        potential_winning_columns = set()

        # Check horizontal "3 in a row"
        for row in range(6):
            for col_start in range(7 - 3):  # Only consider ranges where "3 in a row" is possible
                line = board[row, col_start:col_start + 4]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_col = col_start + np.where(line == 0)[0][0]
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Check vertical "3 in a row"
        for col in range(7):
            for row_start in range(6 - 3):  # Only consider ranges where "3 in a row" is possible
                line = board[row_start:row_start + 4, col]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_row = row_start + np.where(line == 0)[0][0]
                    if env.is_valid_action(col):
                        potential_winning_columns.add(col)

        # Check diagonal (top-left to bottom-right)
        for row_start in range(6 - 3):
            for col_start in range(7 - 3):
                line = [board[row_start + i, col_start + i] for i in range(4)]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_idx = np.where(np.array(line) == 0)[0][0]
                    empty_col = col_start + empty_idx
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Check diagonal (bottom-left to top-right)
        for row_start in range(6 - 3):
            for col_start in range(3, 7):
                line = [board[row_start + i, col_start - i] for i in range(4)]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_idx = np.where(np.array(line) == 0)[0][0]
                    empty_col = col_start - empty_idx
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Ensure at least two distinct columns exist
        if len(potential_winning_columns) >= 2:
            logging.debug(f"Player {current_player} has two '3 in a row' patterns! Winning columns: {list(potential_winning_columns)}")
            return True

        logging.debug(f"No double '3 in a row' patterns detected for Player {current_player}.")
        return False

    def is_WinBlock_move(self, env, last_move_col, current_player):
        """
        Check if the last move by the opponent resulted in a potential winning or blocking scenario.
        This evaluates the current board after the last move has been made.

        Args:
            env (Connect4): The current game environment.
            last_move_col (int): The column where the last move was made.
            current_player (int): The player to analyze the board for (1 or 2).

        Returns:
            bool: True if the last move created a winning or blocking opportunity, False otherwise.
        """
        board = env.get_board()
        opponent = 3 - current_player

        # Check horizontal, vertical, and diagonal lines around the last move
        last_row = next((row for row in range(6) if board[row, last_move_col] == opponent), None)

        if last_row is None:
            logging.debug(f"No piece found in column {last_move_col} to analyze for win/block.")
            return False

        # Check horizontal win/block
        for col_start in range(max(0, last_move_col - 3), min(7, last_move_col + 1)):
            line = board[last_row, col_start:col_start + 4]
            if np.count_nonzero(line == opponent) == 3 and np.count_nonzero(line == 0) == 1:
                logging.debug(f"Horizontal win/block detected at column {col_start + np.where(line == 0)[0][0]}.")
                return True

        # Check vertical win/block
        if last_row <= 2:  # Only check if enough rows below for a vertical line
            line = board[last_row:last_row + 4, last_move_col]
            if np.count_nonzero(line == opponent) == 3 and np.count_nonzero(line == 0) == 1:
                logging.debug(f"Vertical win/block detected at column {last_move_col}.")
                return True

        # Check diagonal (top-left to bottom-right)
        for offset in range(-3, 1):
            diagonal = [board[last_row + i, last_move_col + i] for i in range(4) 
                        if 0 <= last_row + i < 6 and 0 <= last_move_col + i < 7]
            if len(diagonal) == 4 and np.count_nonzero(diagonal == opponent) == 3 and np.count_nonzero(diagonal == 0) == 1:
                logging.debug(f"Diagonal win/block detected at column {last_move_col + np.where(np.array(diagonal) == 0)[0][0]}.")
                return True

        # Check diagonal (top-right to bottom-left)
        for offset in range(-3, 1):
            diagonal = [board[last_row + i, last_move_col - i] for i in range(4) 
                        if 0 <= last_row + i < 6 and 0 <= last_move_col - i < 7]
            if len(diagonal) == 4 and np.count_nonzero(diagonal == opponent) == 3 and np.count_nonzero(diagonal == 0) == 1:
                logging.debug(f"Diagonal win/block detected at column {last_move_col - np.where(np.array(diagonal) == 0)[0][0]}.")
                return True

        logging.debug("No win/block detected after last move.")
        return False

