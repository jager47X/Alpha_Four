import math
import random
import logging
import os
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------- Global Configuration ---------------- #
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.001

EPSILON = 1.0
EPSILON_DECAY = 0.999991
EPSILON_MIN = 0.0001

REPLAY_BUFFER_SIZE = 1000

TARGET_EVALUATE = 100
TARGET_UPDATE = 100
NUM_EPISODES = 100000

RAND_EPISODE_BY = 20000
MCTS_EPISODE_BY = 50000
SELF_LEARN_START = 50000

EVAL_FREQUENCY = 1000

# ------------- Device Setup ------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ Disk-based Replay Buffer ------------------ #
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

        # Create memmap files
        self.states = np.memmap(
            f"{prefix_path}_states.dat",
            dtype=np.float32,
            mode="w+",
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


# ------------------ Connect4 Environment ------------------ #
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

    def make_move(self, action, warning=1):
        if not self.is_valid_action(action) and warning == 1:
            logging.error("Invalid action!")
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.current_player = 3 - self.current_player  # Switch player

    def changeTurn(self):
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Check horizontal, vertical, diagonal
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


# ------------------ DQN Model ------------------ #
class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 6 * 7, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)  # 7 possible actions

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, 6, 7) => we add channel dimension
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten

        # Note: BatchNorm1d can break on a single sample, so guard:
        x = F.relu(self.bn4(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = F.relu(self.bn6(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = self.dropout2(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # final Q-values for 7 actions

        return x


# ------------------ MCTS Implementation ------------------ #
class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = deepcopy(state)
        self.parent = parent
        self.action_taken = action_taken
        self.visits = 0
        self.wins = 0
        self.children = []
        # note: the node holds the "player who just played"
        self.player = 3 - self.state.current_player

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)


def run_mcts(env, num_simulations=100, debug=False):
    root = MCTSNode(env)

    for sim in range(1, num_simulations + 1):
        node = root

        # Selection
        while node.children:
            node = max(node.children, key=lambda c: c.ucb())

        # Expansion
        winner_now = node.state.check_winner()
        if winner_now == 0 and not node.state.is_draw():
            valid_actions = node.state.get_valid_actions()
            for act in valid_actions:
                new_state = deepcopy(node.state)
                new_state.make_move(act)
                child = MCTSNode(new_state, parent=node, action_taken=act)
                node.children.append(child)

            # Randomly pick one child for rollout
            if node.children:
                node = random.choice(node.children)
                winner_now = node.state.check_winner()

        # Simulation (random playout)
        sim_state = deepcopy(node.state)
        while sim_state.check_winner() == 0 and not sim_state.is_draw():
            vacts = sim_state.get_valid_actions()
            if not vacts:
                break
            choice = random.choice(vacts)
            sim_state.make_move(choice)

        final_winner = sim_state.check_winner()

        # Backpropagation
        current = node
        while current is not None:
            current.visits += 1
            if final_winner == current.player:
                current.wins += 1
            elif final_winner == 0:
                current.wins += 0.5  # half-win for draw
            else:
                current.wins -= 1
            current = current.parent

    if not root.children:
        valid_actions = env.get_valid_actions()
        if valid_actions:
            return random.choice(valid_actions)
        return None

    # pick child with highest visits
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action_taken


def parallel_mcts_search(args):
    env, num_simulations, debug = args
    return run_mcts(env, num_simulations, debug)


def run_parallel_mcts(env, total_simulations, num_processes, debug=False):
    simulations_per_process = total_simulations // num_processes
    remaining_simulations = total_simulations % num_processes
    args = [(deepcopy(env), simulations_per_process, debug) for _ in range(num_processes)]
    if remaining_simulations > 0:
        args.append((deepcopy(env), remaining_simulations, debug))
    try:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(parallel_mcts_search, args)
    except Exception as e:
        logging.error(f"Multiprocessing pool encountered an error: {e}")
        return None

    action_counts = {}
    for action in results:
        if action is not None:
            action_counts[action] = action_counts.get(action, 0) + 1

    if not action_counts:
        return random.choice(env.get_valid_actions()) if env.get_valid_actions() else None

    best_action = max(action_counts, key=action_counts.get)
    return best_action


class Parallel_MCTS:
    def __init__(self, num_simulations=100, debug=False, num_processes=4):
        self.num_simulations = num_simulations
        self.debug = debug
        self.num_processes = num_processes

    def select_action(self, env):
        return run_parallel_mcts(env, self.num_simulations, self.num_processes, self.debug)


class MCTS:
    def __init__(self, num_simulations=100, num_processes=4, debug=False):
        self.mcts = Parallel_MCTS(num_simulations=num_simulations, debug=debug, num_processes=num_processes)

    def pick_action(self, env):
        return self.mcts.select_action(env)


# ------------------ Action / Reward Systems ------------------ #
class RewardSystem:
    """
    Calculates rewards based on game state and actions.
    """
    def calculate_reward(self, env, last_action, current_player):
        board = env.get_board()
        opponent = 3 - current_player

        winner = env.check_winner()
        if winner == current_player:
            result_reward = 10.0
            win_status = 1
        elif winner == opponent:
            result_reward = -10.0
            win_status = 2
        elif env.is_draw():
            result_reward = 5.0
            win_status = -1
        else:
            result_reward = 0.0
            win_status = 0

        active_reward = self.get_active_reward(board, last_action, current_player, env)
        passive_penalty = self.get_passive_penalty(board, opponent)
        total_reward = result_reward + active_reward - passive_penalty
        return (total_reward, win_status)

    def get_active_reward(self, board, last_action, current_player, env):
        active_reward = 0
        row_played = self.get_row_played(board, last_action)
        if row_played is None:
            return 0.0

        # Double Threat
        if self.is_double_threat(board, row_played, current_player):
            return 8.0

        # Blocking opponent's 4 in a row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 4):
            return 4.0

        # Creating 3 in a row
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 3):
            return 2.0

        # Blocking opponent's 3 in a row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 3):
            return 1.5

        # Creating 2 in a row
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 2):
            return 0.5

        # Blocking opponent's 2 in a row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 2):
            return 0.3

        # If none of the above triggers, give a small "non-zero" reward
        return 0.05

    def get_passive_penalty(self, board, opponent):
        two_in_a_rows = self.count_n_in_a_row(board, opponent, 2)
        three_in_a_rows = self.count_n_in_a_row(board, opponent, 3)
        return two_in_a_rows * 0.3 + three_in_a_rows * 1.5

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
        for row in range(rows - 1, -1, -1):
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
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r, c+1] == board[r, c+2] == board[r, c+3]):
                    return True
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (board[r, c] == player == board[r+1, c] == board[r+2, c] == board[r+3, c]):
                    return True
        # Diagonal \
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]):
                    return True
        # Diagonal /
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


class ActionSystem:
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.reward_system = RewardSystem()

    def check_immediate_win(self, env, player):
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == player:
                return col
        return None

    def check_immediate_block(self, env, player):
        opponent = 3 - player
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == opponent:
                return col
        return None

    def pick_action(self, env, current_player, epsilon=0.1, episode=1):
        # 1) forced win
        move = self.check_immediate_win(env, current_player)
        if move is not None:
            return move

        # 2) forced block
        move = self.check_immediate_block(env, current_player)
        if move is not None:
            return move

        # 3) Epsilon-random
        if random.random() < epsilon:
            valid_actions = env.get_valid_actions()
            if valid_actions:
                return random.choice(valid_actions)

        # 4) Switch MCTS sims based on episode
        if episode < RAND_EPISODE_BY:
            sims = 10
        else:
            sims = 2000

        num_processes = mp.cpu_count()
        mcts_evaluator = MCTS(num_simulations=sims, num_processes=num_processes, debug=False)
        return mcts_evaluator.pick_action(env)

    def pick_action_dqn(self, env):
        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.policy_net.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        valid_actions = env.get_valid_actions()

        # pick the best among valid
        best_a = None
        best_q = -99999
        for a in valid_actions:
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_a = a

        if best_a is None and valid_actions:
            best_a = random.choice(valid_actions)  # fallback
        return best_a

    def compute_reward(self, env, last_action, current_player):
        return self.reward_system.calculate_reward(env, last_action, current_player)


# ------------------ Training Utilities ------------------ #
def normalize_q_values(q_values):
    if isinstance(q_values, np.ndarray):
        q_values = torch.tensor(q_values, dtype=torch.float32)
    return torch.softmax(q_values, dim=0)


def train_agent(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"Checkpoint file path: {model_path} verified.")
            if 'model_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                logger.info(f"Loaded model from {model_path}, starting from episode {start_episode}.")
            else:
                # if it's a raw state_dict without the top-level dict
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
    policy_net_1, target_net_1,
    policy_net_2, target_net_2,
    optimizer_1, optimizer_2,
    TRAINER_SAVE_PATH, MODEL_SAVE_PATH,
    EPSILON, EPSILON_MIN, EPSILON_DECAY,
    TARGET_UPDATE, logger
):
    try:
        # Update target networks
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        # Save checkpoints
        if episode % TARGET_UPDATE == 0:
            save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
            save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
            logger.info(f"Models saved at episode {episode}")
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")

    return EPSILON


# ------------------ GLOBALS so load_model() can modify them ------------------ #
policy_net_1 = None
target_net_1 = None
optimizer_1 = None
start_ep_1 = 0
replay_buffer_1 = None

policy_net_2 = None
target_net_2 = None
optimizer_2 = None
start_ep_2 = 0
replay_buffer_2 = None

eval_policy_net_1 = None  # For evaluation
eval_target_net_1 = None  # For evaluation


def load_model():
    """
    Load both the 'trainer' model and the 'main' model from disk
    into the global variables. This can be called inside your training loop
    whenever you want to refresh from disk.
    """
    global policy_net_1, target_net_1, optimizer_1, replay_buffer_1, start_ep_1
    global policy_net_2, target_net_2, optimizer_2, replay_buffer_2, start_ep_2
    logger = logging.getLogger()

    # We assume the replay buffers were created already. If not, create them:
    if replay_buffer_1 is None:
        replay_buffer_1 = DiskReplayBuffer(
            capacity=REPLAY_BUFFER_SIZE,
            state_shape=(6, 7),
            prefix_path="trainer_buffer",
            device=device
        )
    if replay_buffer_2 is None:
        replay_buffer_2 = DiskReplayBuffer(
            capacity=REPLAY_BUFFER_SIZE,
            state_shape=(6, 7),
            prefix_path="model_buffer",
            device=device
        )

    policy_net_1, target_net_1, optimizer_1, replay_buffer_1, start_ep_1 = load_model_checkpoint(
        TRAINER_SAVE_PATH,
        policy_net_1, target_net_1, optimizer_1,
        replay_buffer_1, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )

    policy_net_2, target_net_2, optimizer_2, replay_buffer_2, start_ep_2 = load_model_checkpoint(
        MODEL_SAVE_PATH,
        policy_net_2, target_net_2, optimizer_2,
        replay_buffer_2, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )

    logger.info("load_model() completed")


# ------------------- Opponent/Agent Classes for Evaluation ------------------- #
class DQNAgent:
    """
    Minimal DQN-based agent for evaluation.
    Loads a saved DQN model from model_path and uses it to pick actions.
    """
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = DQN(device=self.device).to(self.device)
        # Attempt to load from model_path (if exists)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # possibly raw state_dict
                self.model.load_state_dict(checkpoint)
        self.model.eval()

    def pick_action(self, env):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)  # shape [7]
        # pick best among valid
        best_action = None
        best_q = float('-inf')
        for action in valid_actions:
            if q_values[action].item() > best_q:
                best_q = q_values[action].item()
                best_action = action
        return best_action


class RandomOpponent:
    """
    Always picks a random valid action.
    """
    def pick_action(self, env):
        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions) if valid_actions else None


class MCTSAgent:
    """
    Simple MCTS-based agent that uses run_mcts() with a given number of simulations.
    """
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def pick_action(self, env):
        return run_mcts(env, num_simulations=self.num_simulations)


# ------------------- Evaluation Logic ------------------- #
def play_one_game(agent1, agent2, env=None):
    """
    agent1 starts first, agent2 second.
    Return the winner: 1, 2, or 0 for draw.
    """
    if env is None:
        env = Connect4()
    env.reset()

    while True:
        # agent1 move
        action_1 = agent1.pick_action(env)
        if action_1 is None:
            return 0  # no valid moves => treat as draw
        env.make_move(action_1)
        winner = env.check_winner()
        if winner != 0:
            return winner
        if env.is_draw():
            return 0

        # agent2 move
        action_2 = agent2.pick_action(env)
        if action_2 is None:
            return 0
        env.make_move(action_2)
        winner = env.check_winner()
        if winner != 0:
            return winner
        if env.is_draw():
            return 0


def evaluate_agents(agent1, agent2, n_episodes=100):
    """
    Plays agent1 vs agent2 for n_episodes.
    Returns (agent1_win_rate, agent2_win_rate, draw_rate).
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    for _ in range(n_episodes):
        winner = play_one_game(agent1, agent2)
        if winner == 1:
            agent1_wins += 1
        elif winner == 2:
            agent2_wins += 1
        else:
            draws += 1

    return (agent1_wins / n_episodes, agent2_wins / n_episodes, draws / n_episodes)


def main_evaluation(
    model_path="Connect4_Agent_Model1.pth",
    other_model_path="Connect4_Agent_Model2.pth",
    num_games=100
):
    """
    1) Evaluate loaded model vs Random
    2) Evaluate loaded model vs MCTS agent
    3) Evaluate loaded model vs another DQN model
    """
    # 1) Load main DQN Agent
    agent_main = DQNAgent(model_path, device=device)

    # 2) Evaluate vs Random
    random_opp = RandomOpponent()
    a1_wr, a2_wr, dr = evaluate_agents(agent_main, random_opp, n_episodes=num_games)
    print(f"[VS Random] Agent Win Rate={a1_wr*100:.1f}%, "
          f"Random Opponent Win Rate={a2_wr*100:.1f}%, Draw Rate={dr*100:.1f}%")

    # 3) Evaluate vs MCTS
    mcts_opp = MCTSAgent(num_simulations=200)
    a1_wr, a2_wr, dr = evaluate_agents(agent_main, mcts_opp, n_episodes=num_games)
    print(f"[VS MCTS ] Agent Win Rate={a1_wr*100:.1f}%, "
          f"MCTS Opponent Win Rate={a2_wr*100:.1f}%, Draw Rate={dr*100:.1f}%")

    # 4) Evaluate vs another DQN model
    agent_other = DQNAgent(other_model_path, device=device)
    a1_wr, a2_wr, dr = evaluate_agents(agent_main, agent_other, n_episodes=num_games)
    print(f"[VS Other Model] Agent1:{model_path} Win Rate={a1_wr*100:.1f}%, "
          f"Agent2:{other_model_path} Win Rate={a2_wr*100:.1f}%, Draw Rate={dr*100:.1f}%")


if __name__ == "__main__":
    main_evaluation(
        model_path="Connect4_Agent_Model1.pth",
        other_model_path="Connect4_Agent_Model2.pth",
        num_games=10
    )
