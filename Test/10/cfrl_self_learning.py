import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.optim as optim
import logging
import os

# ------------- Memmap-based Replay Buffer ------------- #
class DiskReplayBuffer:
    """
    A Replay Buffer that stores data on disk via NumPy memmap files.
    This helps avoid running out of CPU/GPU RAM for large buffers.
    """
    def __init__(
        self,
        capacity: int,
        state_shape=(6, 7),
        prefix_path="replay_buffer",
        device="cpu"
    ):
        """
        Args:
            capacity (int): Max number of transitions to store.
            state_shape (tuple): Shape of each state (6,7) for Connect4.
            prefix_path (str): Prefix for the .dat files on disk.
            device (str): 'cpu' or 'cuda'.
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.ptr = 0
        self.full = False

        # Create memmap files for states, next_states, actions, rewards, dones
        self.states = np.memmap(
            f"{prefix_path}_states.dat",
            dtype=np.float32,
            mode="w+",  # Overwrite on each run. Use "r+" to resume existing.
            shape=(capacity, *state_shape),
        )
        self.next_states = np.memmap(
            f"{prefix_path}_next_states.dat",
            dtype=np.float32,
            mode="w+",
            shape=(capacity, *state_shape),
        )
        self.actions = np.memmap(
            f"{prefix_path}_actions.dat",
            dtype=np.int32,
            mode="w+",
            shape=(capacity,),
        )
        self.rewards = np.memmap(
            f"{prefix_path}_rewards.dat",
            dtype=np.float32,
            mode="w+",
            shape=(capacity,),
        )
        self.dones = np.memmap(
            f"{prefix_path}_dones.dat",
            dtype=np.bool_,
            mode="w+",
            shape=(capacity,),
        )

    def push(self, state, action, reward, next_state, done):
        """Store one transition. Overwrites oldest if at capacity."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Advance pointer
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size):
        """Sample a random batch from the buffer, returning torch Tensors."""
        max_idx = self.capacity if self.full else self.ptr
        if batch_size > max_idx:
            raise ValueError(f"Not enough samples: have {max_idx}, need {batch_size}.")

        idxs = np.random.choice(max_idx, batch_size, replace=False)

        states_batch = self.states[idxs]
        actions_batch = self.actions[idxs]
        rewards_batch = self.rewards[idxs]
        next_states_batch = self.next_states[idxs]
        dones_batch = self.dones[idxs]

        states_tensor = torch.tensor(states_batch, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states_batch, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.bool, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor


# ------------- Hyperparameters ------------- #
BATCH_SIZE = 1
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 10000
 
TARGET_UPDATE = 1
NUM_EPISODES = 100000

MODEL_SAVE_PATH = 'Connect4_Agent_Model.pth'
TRAINER_SAVE_PATH = 'Connect4_Agent_Trainer.pth'

# ------------- Device Setup ------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------- Connect4 Environment ------------- #
class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
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

    def make_move(self, action, warning):
        if not self.is_valid_action(action) and warning == 1:
            logging.error("Invalid action!")
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.changeTurn()  # Switch player

    def changeTurn(self):
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Check horizontal, vertical, and diagonal
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

        return 0  # No winner

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [col for col in range(7) if self.is_valid_action(col)]

    def copy(self):
        copied_env = Connect4()
        copied_env.board = self.board.copy()
        copied_env.current_player = self.current_player
        return copied_env


# ------------- DQN Model ------------- #
class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = self.activation(self.bn1(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.bn2(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = self.activation(self.bn3(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = self.dropout2(x)
        x = self.activation(self.bn4(self.fc4(x)) if x.size(0) > 1 else self.fc4(x))
        x = self.activation(self.bn5(self.fc5(x)) if x.size(0) > 1 else self.fc5(x))
        x = self.dropout3(x)
        x = self.activation(self.bn6(self.fc6(x)) if x.size(0) > 1 else self.fc6(x))
        return self.fc7(x)
# ------------- TrainerLogic ------------- #
class TrainerLogic:
    # get Next move
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
    # get Next move
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

    def monte_carlo_tree_search(self, env, num_simulations=1000):
        # Simplified version from your code to fit here
        class MCTSNode:
            def __init__(self, state, parent=None):
                self.state = state.copy()
                self.parent = parent
                self.children = []
                self.visits = 0
                self.wins = 0

            def ucb_score(self, c=1.414):
                if self.visits == 0:
                    return float('inf')
                return (self.wins / self.visits) + c * np.sqrt(np.log(self.parent.visits) / self.visits)

        root = MCTSNode(env)

        for sim in range(num_simulations):
            node = root
            # Selection
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score())
            # Expansion
            if not node.children and not node.state.check_winner():
                valid_actions = node.state.get_valid_actions()
                for move in valid_actions:
                    temp_env = node.state.copy()
                    temp_env.make_move(move, 0)
                    node.children.append(MCTSNode(temp_env, parent=node))
            # Simulation
            if not node.children:
                continue
            current_state = node.state.copy()
            while not current_state.check_winner() and not current_state.is_draw():
                valid_actions = current_state.get_valid_actions()
                if not valid_actions:
                    break
                move = random.choice(valid_actions)
                current_state.make_move(move, 0)
            winner = current_state.check_winner()
            # Backprop
            current_node = node
            while current_node is not None:
                current_node.visits += 1
                if winner == 2:
                    current_node.wins += 1
                elif winner == 1:
                    current_node.wins -= 1
                current_node = current_node.parent

        if not root.children:
            valid_actions = env.get_valid_actions()
            if valid_actions:
                return random.choice(valid_actions)
            raise RuntimeError("Board is full.")

        best_child = max(root.children, key=lambda n: n.visits)
        best_move = None
        for action, child in zip(env.get_valid_actions(), root.children):
            if child == best_child:
                best_move = action
                break
        if best_move is None:
            valid_actions = env.get_valid_actions()
            best_move = random.choice(valid_actions)
        return best_move
        
    def logic_based_action(self, env, current_player,debug=1):
        """
        Use logic to decide the move (winning or blocking).
        If no logical move exists, return None.
        """

        # Check for a winning move for this turn to put
        win_move = self.get_win_move(env, current_player,debug)
        if win_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a winning move in column {win_move}.")
            return win_move

        # Check for a blocking move for this turn to put
        block_move = self.get_block_move(env, current_player,debug)
        if block_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a blocking move in column {block_move}.")
            return block_move

        # No logical move found
        if debug==1:
            logging.debug(f"Player {current_player} found no logical move.")
        return None


    def MCTS_action(self, env, current_episode):
        if current_episode > 10000:
            sims = 10000
        else:
            sims = current_episode
        mcts_action = self.monte_carlo_tree_search(env, sims)
        logging.debug(f"MCTS used level={sims}")
        return mcts_action
# ------------- Agent Logic ------------- #
class AgentLogic:
    def __init__(self, policy_net):
        self.policy_net = policy_net
    
    def getAction(self, env, current_episode):
        # Q-values
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(self.policy_net.device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()

        # Softmax or any normalization
        normalized_q_values = self.normalize_q_values(q_values)
        valid_actions = env.get_valid_actions()
        valid_qs = {a: q_values[a] for a in valid_actions}
        action = max(valid_qs, key=lambda a: valid_qs[a])
        
        logging.debug(f"SoftMaxed Q-Values: {normalized_q_values}")
        return action
    def normalize_q_values(self,q_values):
        if isinstance(q_values, np.ndarray):
            q_values = torch.tensor(q_values, dtype=torch.float32)
        return torch.softmax(q_values, dim=0)

    def getReward(self,env, last_action, current_player):
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
        rs = self.RewardSystem()
        return rs.calculate_reward(env, last_action, current_player)
    # ------------- Reward System ------------- #
    class RewardSystem:
        """
        You may include any class-level documentation or initialization here, if needed.
        """
        
        def calculate_reward(self, env, last_action, current_player):
            """
            Calculate the reward for the given player based on the game state and potential outcomes.
            
            Args:
                env: The current game environment.
                last_action: The action taken by the player.
                current_player: The player (1 or 2) for whom the reward is calculated.  

            Returns:
                tuple: (total_reward, win_status), where:
                    - total_reward (float): active_reward - passive_reward + result_reward
                    - result_reward (float): Based on game status if Current player wins, 
                    then return high reward 10.0, if Current player loses, then return 
                    high penality -10.0 otherwise 1.
                    - active_reward (float): Based on Current player's move
                    - passive_reward (float): Based on Current player's move
                    - win_status (int): 1 if current_player wins, -1 if opponent_player wins, 0 otherwise.
            Main Logic:
                Calculate active reward:
                    - if Current player chooce Win Move (next turn win whatever Opponent choose) then add extreme high reward 5
                    - if Current player chooce Block WinMove (block four in row by adding one block) then add extreme high reward 2.5
                    - if Current player add 3 in row then add mid reward 1.5
                    - if Current player blocks 3 in row the Opponent player then add reward 1.0
                    - if Current player add 2 in row then add reward 0.5
                    - if Current player blocks 2 in row then low reward 0.25
                    - Otherwise add 0.01

                Calculate passive reward:
                    - Calculate each connection of 2 blocks in the board multiply by 0.1 (i.e 2 blocks= 0.2, 3 blocks 0.3)
                    - Calculate each connection of 3 blocks in the board multiply by 1
                    - Otherwise add 0
            """
            board = env.get_board()
            self.opponent = 3 - current_player

            winner = env.check_winner()
            if winner == current_player:
                # Current player just won
                result_reward = 10.0
                win_status = 1
            elif winner == self.opponent:
                # Opponent won
                result_reward = -10.0
                win_status = 2
            elif env.is_draw():
                # draw
                result_reward = 5.0
                win_status=-1
            else: # on going
                win_status = 0
                result_reward = 0
            # Active reward: based on the single move just made
            active_reward = self.get_active_reward(board, last_action, current_player)

            # Passive reward: based on the board state for current_player
            passive_reward = self.get_passive_reward(board, current_player)

            total_reward = result_reward + active_reward - passive_reward
            return (total_reward, win_status)

        # ----------------------------------------------------
        #  Below are helper functions to detect patterns, etc.
        # ----------------------------------------------------
        
        def get_active_reward(self, board, last_action, current_player):
            """
            Determine the reward from the single move the current_player just made.
            Checks for:
                - Win Move (4 in a row)
                - Block Win Move
                - 3 in a row
                - Block 3 in a row
                - 2 in a row
                - Block 2 in a row
                - Otherwise 0.01
            """
            row_played = self.get_row_played(board, last_action)
            if row_played is None:
                logging.debug("No move found in the given column.")
                return 0.0

            # Check immediate 4-in-a-row = "Win Move"
            if self.causes_WinMove(board, row_played, last_action, current_player):
                return 5

            # Check if it blocks the opponent's 4-in-a-row
            if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 4):
                return 2.5

            # Check 3 in a row
            if self.causes_n_in_a_row(board, row_played, last_action, current_player, 3):
                return 1.5

            # Check if it blocks opponent's 3 in a row
            if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 3):
                return 1

            # Check 2 in a row
            if self.causes_n_in_a_row(board, row_played, last_action, current_player, 2):
                return 0.5

            # Check if it blocks opponent's 2 in a row
            if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 2):
                return 0.25

            # Otherwise minimal reward
            return 0.01

        def get_passive_reward(self, board, current_player):
            """
            Calculate a "passive" reward based on the entire board state for current_player:
                - Count total 2-in-a-rows => each adds 0.1
                - Count total 3-in-a-rows => each adds 1.0
            """
            two_in_a_rows = self.count_n_in_a_row(board, current_player, 2)
            three_in_a_rows = self.count_n_in_a_row(board, current_player, 3)

            # 2 in a row => 0.1 per
            # 3 in a row => 1.0 per
            passive_reward = two_in_a_rows * 0.1 + three_in_a_rows * 1.0
            return passive_reward
        
        def get_row_played(self, board, col):
            """
            Returns the row index where the last piece in 'col' is placed.
            If no piece is found, return None.
            """
            rows = board.shape[0]
            for r in range(rows):
                if board[r, col] != 0:  # Found a piece
                    return r
            return None
        def causes_WinMove(board, row_played, last_action, current_player):
            {
            """
            Check for the last_action leads next turn wining no matter how the opponent choose
            run bruteforce
            """
            temp_env = board.copy()
            temp_env = temp_env.changeTurn()
            potenital_win=0
            for potenital_opponent_move in temp_env.get_valid_actions():
                temp_env = env.copy()
                temp_env = temp_env.changeTurn()
                temp_env.make_move(potenital_opponent_move, warning=0)  # Simulate the move
                #logging.debug(temp_env.get_board())
                if temp_env.check_winner() == self.opponent:
                    return False # found win move for opponent
                
                # Check Current Player Wins every single game 
                for potenital_win_move in temp_env.get_valid_actions():
                definiteWinChosen=False   
                temp_env = env.copy()
                temp_env = temp_env.changeTurn()
                temp_env.make_move(potenital_win_move, warning=0)  # Simulate the move
                #Find one winning each iteration or False
                if temp_env.check_winner() == current_player:
                    potenital_win+=1
            if potenital_win==temp_env.get_valid_actions() # Within everysingle valid action this move wins next move
                definiteWinChosen=True

           if definiteWinChosen:
               if debug==1:
                        logging.debug(f"{current_player} caused definite WinMove")
               return True 
           return False
        }
        def causes_n_in_a_row(self, board, row, col, player, n):
            """
            Check if placing at (row, col) caused 'player' to have n in a row.
            """
            return (
                self.check_line(board, row, col, player, n, "horizontal") or
                self.check_line(board, row, col, player, n, "vertical")   or
                self.check_line(board, row, col, player, n, "diag1")      or
                self.check_line(board, row, col, player, n, "diag2")
            )

        def blocks_opponent_n_in_a_row(self, board, row, col, current_player, n):
            """
            Temporarily remove current_player's piece and check if 
            that spot would have caused the opponent to get 'n' in a row.
            """
            
            original_value = board[row, col]
            board[row, col] = 0
            caused = self.causes_n_in_a_row(board, row, col, self.opponent, n)
            board[row, col] = original_value  # restore
            return caused

        def check_line(self, board, row, col, player, n, direction):
            """
            Check if there's a continuous line of length >= n for 'player' that includes (row, col),
            in the given direction.
            """
            rows, cols = board.shape

            def count_consecutive(r_step, c_step):
                count = 1  # include current cell
                # forward
                rr, cc = row + r_step, col + c_step
                while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                    count += 1
                    rr += r_step
                    cc += c_step
                # backward
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
            elif direction == "diag1":  # top-left -> bottom-right
                return count_consecutive(1, 1) >= n
            elif direction == "diag2":  # top-right -> bottom-left
                return count_consecutive(1, -1) >= n

        def count_n_in_a_row(self, board, player, n):
            """
            Count how many distinct sequences of length >= n for 'player' exist on the board.
            We'll use a simple approach to identify all n-length segments in the four directions
            (horizontal, vertical, and the two diagonals). 
            """
            rows, cols = board.shape
            visited_segments = set()
            total_count = 0

            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            def in_bounds(r, c):
                return 0 <= r < rows and 0 <= c < cols

            # Check each cell for possible lines
            for r in range(rows):
                for c in range(cols):
                    if board[r, c] == player:
                        for dr, dc in directions:
                            cells_in_line = []
                            rr, cc = r, c
                            # Collect continuous sequence in the forward direction
                            while in_bounds(rr, cc) and board[rr, cc] == player:
                                cells_in_line.append((rr, cc))
                                rr += dr
                                cc += dc
                            # If the length is at least n, we might have multiple overlapping segments
                            if len(cells_in_line) >= n:
                                # Slide a window of length n across the line
                                for start_idx in range(len(cells_in_line) - n + 1):
                                    segment = tuple(cells_in_line[start_idx:start_idx + n])
                                    if segment not in visited_segments:
                                        visited_segments.add(segment)
                                        total_count += 1
            return total_count 


    
# ------------- Training Utilities ------------- #

def train_agent(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return  # not enough data

    # Sample from disk-based replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Current Q
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Target Q
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones.float()) * GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_log_file_name():
    log_num = 1
    while os.path.exists(f"log{log_num}.txt"):
        log_num += 1
    return f"log{log_num}.txt"

def setup_logger(log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def load_model_checkpoint(model_path, policy_net, target_net, optimizer,
                          replay_buffer, learning_rate, buffer_size, logger, device):
    try:
        if policy_net is None:
            policy_net = DQN().to(device)
        if target_net is None:
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
        if optimizer is None:
            optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        if replay_buffer is None:
            # Not used here since we use DiskReplayBuffer
            pass

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"Checkpoint file path: {model_path} verified.")
            if 'model_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                logger.info(f"Loaded model from {model_path}, starting from episode {start_episode}.")
            else:
                policy_net.load_state_dict(checkpoint)
                start_episode = 0
                logger.info(f"Loaded raw state_dict from {model_path}. Starting from episode {start_episode}.")
        else:
            logger.error(f"Checkpoint file {model_path} does not exist. Starting fresh.")
            start_episode = 0

    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh.")
        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
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

def periodic_updates(
    episode,
    policy_net_2, target_net_2,
    optimizer_2,
    MODEL_SAVE_PATH,
    EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
):
    try:
        if episode % TARGET_UPDATE == 0:
            #target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        if episode % TARGET_UPDATE == 0:
            #save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
            save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
            logger.info(f"Models saved at episode {episode}")
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON


# ------------- Main Training Loop ------------- #
def main():
    global EPSILON, NUM_EPISODES

    log_file_name = get_log_file_name()
    logger = setup_logger(log_file_name, logging.DEBUG)
    print("Starting Initialization")

    # Instead of a deque, set up two DiskReplayBuffers
 
    replay_buffer = DiskReplayBuffer(
        capacity=REPLAY_BUFFER_SIZE,
        state_shape=(6, 7),
        prefix_path="model_buffer",
        device=device
    )

    # Load or initialize the network (policy and target for each agent)
    policy_net, target_net, optimizer, _, start_ep = load_model_checkpoint(
        MODEL_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )

    current_episode = start_ep
    NUM_EPISODES -= current_episode
    logger.info(f"NUM_EPISODES adjusted to {NUM_EPISODES} after subtracting {current_episode}.")

    trainer_wins, agent_wins, draws = 0, 0, 0
    env = Connect4()
    print("Starting Initialization is over, now training.")
    for episode in range(1, NUM_EPISODES + 1):
        logger.info(f"==== Starting episode {episode} / {NUM_EPISODES} ====")
        state = env.reset()
        done = False
        trainer_logic= TrainerLogic()
        agent_logic = AgentLogic(policy_net)
        total_reward=0
        while not done:
            # 1) Agent 1's turn
            action_1 = trainer_logic.logic_based_action(env, 1)
            if action_1 is not None:
                logging.debug(f"Trainer: Logic-based action (win/block): {action_1}")
            else:
                action_1 = trainer_logic.MCTS_action(env, episode)
                logging.debug(f"Trainer: MCTS Method: {action_1}")
            env.make_move(action_1, 1)
            win_status = env.check_winner()
            done = (win_status != 0)
            
            # 2) Agent 2's turn (only if not done)
            if not done:
                if EPSILON > random.random():
                    valid_actions = env.get_valid_actions()
                    action_2 = random.choice(valid_actions)
                    logging.debug(f"Agent: Random Choice: {action_2}")
                else:
                    action_2 = agent_logic.getAction(env, episode)
                    logging.debug(f"Agent: DQN Method: {action_2}")
                env.make_move(action_2, 1)
                reward, win_status = agent_logic.getReward(env, action_2, current_player=2)
                total_reward+=reward
        
                next_state = env.board.copy()
                done = (win_status != 0)

                # Push to disk-based buffer for agent_2
                replay_buffer.push(state, action_2, reward, next_state, done)
                state = next_state

        # After the episode ends, train the agent
        train_agent(policy_net, target_net, optimizer, replay_buffer)

        # Periodic updates
        EPSILON = periodic_updates(
            episode,
            policy_net, target_net,
            optimizer,
            MODEL_SAVE_PATH,
            EPSILON, EPSILON_MIN, EPSILON_DECAY,
            TARGET_UPDATE, logger
        )

        # Check final outcome for logging
        if win_status == 1:
            trainer_wins += 1
            winner_str = "Trainer Won"
        elif win_status == 2:
            agent_wins += 1
            winner_str = "Agent Won"
        else:
            draws += 1
            winner_str = "Draw"
        print(
            f"Episode {episode} / {NUM_EPISODES} Game Status:{winner_str}. "
            f"(Trainer wins={trainer_wins}, Agent wins={agent_wins} Total Reward:{total_reward}, draws={draws})"
        )
        logger.info(
            f"Episode {episode} / {NUM_EPISODES} Game Status:{winner_str}. "
            f"(Trainer wins={trainer_wins}, Agent wins={agent_wins} Total Reward:{total_reward}, draws={draws})"
        )

    logger.info("Training complete")

if __name__ == "__main__":
    main()
