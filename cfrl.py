import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.optim as optim
#from connect4 import Connect4  # Import the Connect4 class
import logging
import random
import os
import sys
# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 1000000

# Define paths to save and load models in Google Drive
MODEL_SAVE_PATH = 'Connect4_Agent_Model2.pth'
TRAINER_SAVE_PATH = 'Connect4_Agent_Trainer2.pth'
# Set up devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 6 rows, 7 columns
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board
    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.board.shape[1]):
            logging.error(f"Action {action} is out of bounds.")
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action,warning):
        if not self.is_valid_action(action) and warning==1:
            logging.error("Invalid action!")
        for row in range(5, -1, -1):  # Start from the bottom row
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
    def changeTurn(self):
        self.current_player = 3 - self.current_player
    def check_winner(self):
        # Check horizontal, vertical, and diagonal for a win
        for row in range(6):
            for col in range(7 - 3):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row, col:col + 4] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(7):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row:row + 4, col] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(7 - 3):
                if self.board[row, col] != 0 and \
                   np.all([self.board[row + i, col + i] == self.board[row, col] for i in range(4)]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(3, 7):
                if self.board[row, col] != 0 and \
                   np.all([self.board[row + i, col - i] == self.board[row, col] for i in range(4)]):
                    return self.board[row, col]

        return 0  # No winner yet

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [col for col in range(7) if self.is_valid_action(col)]
    
    def copy(self):
        """Return a deep copy of the current environment state."""
        copied_env = Connect4()
        copied_env.board = self.board.copy()
        copied_env.current_player = self.current_player
        return copied_env

"""#DQN.py"""



class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Layer definitions
        self.fc1 = nn.Linear(6 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 7)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 6 * 7)  # Flatten the input

        # Layer forward passes with condition for batch normalization
        x = self.activation(self.bn1(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = self.activation(self.bn2(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = self.activation(self.bn3(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = self.dropout2(x)  # Apply dropout
        x = self.activation(self.bn4(self.fc4(x)) if x.size(0) > 1 else self.fc4(x))
        x = self.activation(self.bn5(self.fc5(x)) if x.size(0) > 1 else self.fc5(x))
        x = self.dropout3(x)  # Apply dropout
        x = self.activation(self.bn6(self.fc6(x)) if x.size(0) > 1 else self.fc6(x))
        return self.fc7(x)

"""#AgentLogic.py

#Connect4gui.py
"""
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





    def monte_carlo_tree_search(self, env, num_simulations=1000):
        """
        Monte Carlo Tree Search for decision-making in Connect4.
        Returns the column index of the best move.
        """
        class MCTSNode:
            def __init__(self, state, parent=None):
                self.state = state.copy()
                self.parent = parent
                self.children = []
                self.visits = 0
                self.wins = 0

            def ucb_score(self, exploration_constant=1.414):
                if self.visits == 0:
                    return float('inf')
                win_rate = self.wins / self.visits
                exploration_term = exploration_constant * \
                    (np.sqrt(np.log(self.parent.visits) / self.visits))
                return win_rate + exploration_term

        root = MCTSNode(env)
        logging.debug(f"Starting MCTS with {num_simulations} simulations.")

        for _ in range(num_simulations):
            node = root

            # Selection: Traverse tree using UCB
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score())
                #logging.debug(f"Selected node with UCB score: {node.ucb_score()}")

            # Expansion: Create child nodes if not terminal
            if not node.children and not node.state.check_winner():
                valid_actions = node.state.get_valid_actions()
                #logging.debug(f"Expanding node. Valid actions: {valid_actions}")
                for move in valid_actions:
                    temp_env = node.state.copy()
                    logic_based_move =self.logic_based_action(temp_env,temp_env.current_player,0)#check 3 in rows
                    if logic_based_move is not None:
                        move = logic_based_move
                    temp_env.make_move(move, 0)
                    child_node = MCTSNode(temp_env, parent=node)
                    node.children.append(child_node)

            # If no children were created, skip to next simulation
            if not node.children:
                #logging.debug("No children created. Skipping simulation.")
                continue

            # Simulation: Randomly play out the game
            current_state = node.state.copy()
            while not current_state.check_winner() and not current_state.is_draw():
                valid_actions = current_state.get_valid_actions()
                if not valid_actions:
                    #logging.debug("No valid actions available during simulation.")
                    break
                logic_based_move =self.logic_based_action(current_state,temp_env.current_player,0)#check 3 in rows
                if logic_based_move is not None:
                        move = logic_based_move
                else:
                    move = random.choice(valid_actions)
                current_state.make_move(move, 0)

            # Backpropagation: Update visits and wins
            winner = current_state.check_winner()
            current_node = node
            while current_node is not None:
                current_node.visits += 1
                if winner == 2:  # Adjust for current player
                    current_node.wins += 1
                elif winner == 1:
                    current_node.wins -= 1
                current_node = current_node.parent
            {current_simulation}
            logging.debug(f"process:{current_simulation}/{num_simulations} ")

        # Ensure children exist before selecting the best move
        if not root.children:
            #logging.warning("MCTS failed to generate children nodes. Falling back to random valid action.")
            valid_actions = env.get_valid_actions()
            if valid_actions:
                logic_based_move =self.logic_based_action(env,temp_env.current_player,0)#check 3 in rows
                if logic_based_move is not None:
                    return logic_based_move
                return random.choice(valid_actions)
            else:
                logging.critical("No valid actions available.")
                raise RuntimeError("Board is already full.")

        # Select the best child node based on visits
        best_child = max(root.children, key=lambda n: n.visits)

        # Find the corresponding action for the best child node
        best_move = None
        for action, child in zip(env.get_valid_actions(), root.children):
            if child == best_child:
                best_move = action
                break

        if best_move is None:
            logging.warning("Failed to find the best move. Falling back to random valid action.")
            best_move = random.choice(env.get_valid_actions())

        logging.debug(f"Best move selected: {best_move} with {best_child.visits} visits.")
        return best_move



    def combined_action(self, env):
        """
        Decide the action for the AI (Player 2) using logical rules, MCTS, and DQN.
        """
        current_player = env.current_player
        device = self.policy_net.device  # Access the device from the policy_net
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()
        q_values =normalize_q_values(q_values)
        valid_actions = env.get_valid_actions()
        valid_q_values = {action: q_values[action] for action in valid_actions}
        
        formatted_q_values = {action: f"{q_value:.3f}" for action, q_value in valid_q_values.items()}
        logging.debug(f"Q-values: {formatted_q_values}")
        # Select the action with the highest Q-value
        action = max(valid_q_values, key=lambda a: valid_q_values[a])
        max_q_value = valid_q_values[action]
        
        if max_q_value < 0.5 or EPSILON > 0.5:
            # Use logic-based or MCTS if Q-values are too low or begging of stage
            action = self.monte_carlo_tree_search(env, num_simulations=1000)
            if action is not None:
                logging.debug(f"Player{current_player}: Using MCTS for action: {action}")
                return action
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



# Training function for a single agent
def train_agent(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        logger.info("Not enough data to train")
        return  # Not enough data to train

    # Sample mini-batch from replay buffer
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)


    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    next_q_values = target_net(next_states).max(1)[0]
    targets = rewards + (1 - dones.float()) * GAMMA * next_q_values

    # Compute loss and optimize
    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Function to generate unique log file name
def get_log_file_name():
    log_num = 1
    while os.path.exists(f"log{log_num}.txt"):
        log_num += 1
    return f"log{log_num}.txt"

# Set up logging
def setup_logger(log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)

    # Add handler to console as well
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
# Function to load model checkpoints
def load_model_checkpoint(model_path, policy_net, target_net, optimizer, replay_buffer, learning_rate, buffer_size, logger, device):
    try:
        # Ensure networks are initialized
        if policy_net is None:
            logger.info("Initializing policy network...")
            policy_net = DQN().to(device)
        if target_net is None:
            logger.info("Initializing target network...")
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
        if optimizer is None:
            logger.info("Initializing optimizer...")
            optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        if replay_buffer is None:
            logger.info("Initializing replay buffer...")
            replay_buffer = deque(maxlen=buffer_size)

        # Attempt to load the checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"Checkpoint file path: {model_path} verified.")
            # Check if the checkpoint is structured
            if 'model_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                logger.info(f"Loaded model from {model_path}, starting from episode {start_episode}.")
            else:
                policy_net.load_state_dict(checkpoint)
                start_episode = 0  # Raw state dict, no episode info
                logger.info(f"Loaded raw state_dict from {model_path}. Starting from episode {start_episode}.")
        else:
            logger.error(f"Checkpoint file path {model_path} does not exist. Starting fresh training.")
            start_episode = 0

    except Exception as e:
        # Handle loading failure
        logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh training.")
        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = deque(maxlen=buffer_size)
        start_episode = 0

    return policy_net, target_net, optimizer, replay_buffer, start_episode
def save_model(model_path, policy_net, optimizer, current_episode, logger):
    try:
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': current_episode
        }, model_path)
        logger.info(f"Model checkpoint saved to {model_path} at episode {current_episode}.")
    except Exception as e:
        logger.critical(f"Failed to save model checkpoint to {model_path}: {e}")

# Update target networks, decay epsilon, and save models periodically
def periodic_updates(
        episode, policy_net_1, target_net_1, policy_net_2, target_net_2,
                    optimizer_1,optimizer_2,
                     TRAINER_SAVE_PATH, MODEL_SAVE_PATH, EPSILON, EPSILON_MIN, 
                     EPSILON_DECAY, TARGET_UPDATE, logger):
    try:
        # Update target networks periodically
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        # Save models periodically
        if episode % TARGET_UPDATE == 0:
            save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
            save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
            logger.info(f"Models saved at episode {episode}")

    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON


def normalize_q_values(q_values):
    """
    Normalize Q-values to the range [0, 1].
    
    Args:
        q_values (torch.Tensor or np.ndarray): Input Q-values.
        
    Returns:
        torch.Tensor: Normalized Q-values in the range [0, 1].
    """
    if isinstance(q_values, np.ndarray):  # Convert NumPy array to PyTorch tensor
        q_values = torch.tensor(q_values)

    q_min = torch.min(q_values)
    q_max = torch.max(q_values)

    # Avoid division by zero
    if q_max == q_min:
        return torch.ones_like(q_values)  # All normalized values are 1 if Q-values are identical

    return (q_values - q_min) / (q_max - q_min)  # Min-max normalization



# Main function
def main():
    global EPSILON, NUM_EPISODES


    # Load checkpoints for policy_net_1 and policy_net_2
    policy_net_1, target_net_1, optimizer_1, replay_buffer_1, start_episode_1 = load_model_checkpoint(
        TRAINER_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )
    policy_net_2, target_net_2, optimizer_2, replay_buffer_2, start_episode_2 = load_model_checkpoint(
        MODEL_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )
    # Determine NUM_EPISODES based on starting episodes
    current_episode = min(start_episode_1, start_episode_2)
    NUM_EPISODES -= current_episode  # Adjust NUM_EPISODES by subtracting current_episode
    logger.info(f"NUM_EPISODES adjusted to {NUM_EPISODES} after subtracting the current episode.")

    
    agent_1_wins = 0
    agent_2_wins = 0
    draws = 0

    env = Connect4()
    logger.info(f"Starting episode: {current_episode}")

    for episode in range(1, NUM_EPISODES + 1):
        logger.debug(f"Current episode: {episode}")
        state = env.reset()
        done = False
        total_reward_1, total_reward_2 = 0, 0
        agent_logic_1 = AgentLogic(policy_net_1)
        agent_logic_2 = AgentLogic(policy_net_2)

        while not done:
            # Agent 1's turn
            action_1 =agent_logic_1.logic_based_action(env,1)
            if action_1 is not None:
                logging.debug(f"Trainer: Logic-based action (win/block): {action_1}")
            elif EPSILON > random.random():
                valid_actions = env.get_valid_actions()
                logging.debug(f"valid actions{valid_actions}")
                action_1 = random.choice(valid_actions)#error
                logger.debug(f"Trainer: RAND choice:{action_1}")
            else:
                action_1 = agent_logic_1.combined_action(env)  # Use combined_action
                logger.debug(f"Trainer: combinedAction choice:{action_1}")
            
            env.make_move(action_1,1)
            logging.debug("After action_1: board")
            logger.debug(env.get_board())
            # Calculate reward for Agent 1
            reward_1,win_status  = agent_logic_1.calculate_reward(env, action_1, current_player=1)

            next_state = env.board.copy()
            
            replay_buffer_1.append((state, action_1, reward_1, next_state, done))
            state = next_state
            total_reward_1 += reward_1
            logger.debug(f"Trainer: reward: {reward_1}, Total reward: {total_reward_1}")
            done = win_status != 0 or env.is_draw()  # Use win_status to determine game over
            # Agent 2's turn            
            if not done:
                action_2 =agent_logic_2.logic_based_action(env,2)
                if action_2 is not None:
                    logging.debug(f"Trainer: Logic-based action (win/block): {action_2}")
                elif EPSILON > random.random():
                    valid_actions = env.get_valid_actions()
                    logging.debug(f"valid actions{valid_actions}")
                    action_2 = random.choice(valid_actions)
                    logger.debug(f"Agent: RAND choice:{action_2}")
                else:
                    action_2 = agent_logic_2.combined_action(env)  # Use combined_action
                    logger.debug(f"Agent: combinedAction choice:{action_2}")
                env.make_move(action_2,1)
                # Calculate reward for Agent 2
                logging.debug("After action_2: board")
                logger.debug(env.get_board())
            reward_2,win_status = agent_logic_2.calculate_reward(env, action_2, current_player=2)

            next_state = env.board.copy()
            done = win_status!= 0 or env.is_draw()
            replay_buffer_2.append((state, action_2, reward_2, next_state, done))
            state = next_state
            total_reward_2 += reward_2
            logger.debug(f"Agent: reward: {reward_2}, Total reward: {total_reward_2}")
            if done and win_status==2: #evaluation for player 1 after player2 win
                total_reward_1-=1


        # Train both agents
        train_agent(policy_net_1, target_net_1, optimizer_1, replay_buffer_1)
        train_agent(policy_net_2, target_net_2, optimizer_2, replay_buffer_2)
        logger.debug(f"Training completed for episode {episode}")

        # Perform periodic updates
        EPSILON = periodic_updates(
        episode, policy_net_1, target_net_1, policy_net_2, target_net_2,optimizer_1,optimizer_2,
        TRAINER_SAVE_PATH, MODEL_SAVE_PATH, EPSILON, EPSILON_MIN, EPSILON_DECAY,
        TARGET_UPDATE, logger)
        


        # Show the progress every episode
        if env.check_winner() == 1:
            agent_1_wins += 1
            winner = "Agent 1"
        elif env.check_winner() == 2:
            agent_2_wins += 1
            winner = "Agent 2"
        else:
            draws += 1
            winner = "Draw"
        logger.info(f"episode {episode}: Player {winner} wins! (Agent 1 Wins: {agent_1_wins}, Agent 2 Wins: {agent_2_wins}, Draws: {draws})")

    
    logger.info("Training complete")


if __name__ == "__main__":
    log_file_name = get_log_file_name()
    logger = setup_logger(log_file_name,logging.DEBUG)
    main()
