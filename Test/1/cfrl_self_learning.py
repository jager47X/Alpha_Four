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
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 1000
 
TARGET_UPDATE = 100
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
        self.current_player = 3 - self.current_player  # Switch player

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


# ------------- Agent Logic ------------- #
class AgentLogic:
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def get_win_move(self, env, player, debug=1):
        for col in env.get_valid_actions():
            temp_env = env.copy()
            temp_env.make_move(col, warning=0)
            if temp_env.check_winner() == player:
                if debug == 1:
                    logging.debug(f"Player {player} can win by placing in column {col}.")
                return col
        return None

    def get_block_move(self, env, player, debug=1):
        opponent = 3 - player
        for col in env.get_valid_actions():
            temp_env = env.copy()
            temp_env.changeTurn()
            temp_env.make_move(col, warning=0)
            if temp_env.check_winner() == opponent:
                if debug == 1:
                    logging.debug(f"Player {player} can block opponent in column {col}.")
                return col
        return None

    def logic_based_action(self, env, current_player, debug=1):
        # Check winning move
        win_move = self.get_win_move(env, current_player, debug)
        if win_move is not None:
            return win_move
        # Check blocking move
        block_move = self.get_block_move(env, current_player, debug)
        if block_move is not None:
            return block_move
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

    def combined_action(self, env, current_episode):
        # Q-values
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(self.policy_net.device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()

        # Softmax or any normalization
        q_values = normalize_q_values(q_values)
        valid_actions = env.get_valid_actions()
        valid_qs = {a: q_values[a] for a in valid_actions}
        action = max(valid_qs, key=lambda a: valid_qs[a])
        max_q = valid_qs[action]

        # If the max Q is too low, or early in training, do MCTS
        if max_q < 0.5 and EPSILON > 0.1:
            if current_episode > 10000:
                sims = 10000
            else:
                sims = current_episode
            mcts_action = self.monte_carlo_tree_search(env, sims)
            logging.debug(f"MCTS used level={sims}")
            return mcts_action
        logging.debug(f"Confident with Q-Value{max_q}")
        return action

    def calculate_reward(self, env, last_action, current_player):
        opponent = 3 - current_player
        winner = env.check_winner()
        if winner == current_player:
            return 1.0, current_player
        elif winner == opponent:
            return -1.0, opponent
        elif env.is_draw():
            return 0.0, 0
        return -0.01, 0  # default small penalty


# ------------- Training Utilities ------------- #
def normalize_q_values(q_values):
    if isinstance(q_values, np.ndarray):
        q_values = torch.tensor(q_values, dtype=torch.float32)
    return torch.softmax(q_values, dim=0)

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
    policy_net_1, target_net_1,
    policy_net_2, target_net_2,
    optimizer_1, optimizer_2,
    TRAINER_SAVE_PATH, MODEL_SAVE_PATH,
    EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
):
    try:
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        if episode % TARGET_UPDATE == 0:
            save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
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
    replay_buffer_1 = DiskReplayBuffer(
        capacity=REPLAY_BUFFER_SIZE,
        state_shape=(6, 7),
        prefix_path="trainer_buffer",
        device=device
    )
    replay_buffer_2 = DiskReplayBuffer(
        capacity=REPLAY_BUFFER_SIZE,
        state_shape=(6, 7),
        prefix_path="model_buffer",
        device=device
    )

    # Load or initialize the two networks (policy and target for each agent)
    policy_net_1, target_net_1, optimizer_1, _, start_ep_1 = load_model_checkpoint(
        TRAINER_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )
    policy_net_2, target_net_2, optimizer_2, _, start_ep_2 = load_model_checkpoint(
        MODEL_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )

    current_episode = min(start_ep_1, start_ep_2)
    NUM_EPISODES -= current_episode
    logger.info(f"NUM_EPISODES adjusted to {NUM_EPISODES} after subtracting {current_episode}.")

    agent_1_wins, agent_2_wins, draws = 0, 0, 0
    env = Connect4()
    print("Starting Initialization is over, now training.")
    for episode in range(1, NUM_EPISODES + 1):
        logger.info(f"==== Starting episode {episode} / {NUM_EPISODES} ====")
        state = env.reset()
        done = False
        agent_logic_1 = AgentLogic(policy_net_1)
        agent_logic_2 = AgentLogic(policy_net_2)
        total_reward_1=0
        total_reward_2=0
        while not done:
            # 1) Agent 1's turn
            action_1 = agent_logic_1.logic_based_action(env, 1)
            if action_1 is not None:
                logging.debug(f"Trainer: Logic-based action (win/block): {action_1}")
            elif EPSILON > random.random():
                valid_actions = env.get_valid_actions()
                action_1 = random.choice(valid_actions)
                logging.debug(f"Trainer: Random Choice: {action_1}")
            else:
                action_1 = agent_logic_1.combined_action(env, episode)
                logging.debug(f"Trainer: Combined Method: {action_1}")
 
            env.make_move(action_1, 1)
            reward_1, win_status = agent_logic_1.calculate_reward(env, action_1, current_player=1)
            total_reward_1+=reward_1
            next_state = env.board.copy()
            # Push to disk-based buffer for agent_1
            replay_buffer_1.push(state, action_1, reward_1, next_state, done)
            state = next_state
            done = (win_status != 0) or env.is_draw()

            # 2) Agent 2's turn (only if not done)
            if not done:
                action_2 = agent_logic_2.logic_based_action(env, 2)
                if action_2 is not None:
                    logging.debug(f"Agent: Logic-based action (win/block): {action_2}")
                elif EPSILON > random.random():
                    valid_actions = env.get_valid_actions()
                    action_2 = random.choice(valid_actions)
                    logging.debug(f"Agent: Random Choice: {action_2}")
                else:
                    action_2 = agent_logic_2.combined_action(env, episode)
                    logging.debug(f"Agent: Combined Method: {action_2}")
                env.make_move(action_2, 1)
                reward_2, win_status = agent_logic_2.calculate_reward(env, action_2, current_player=2)
                total_reward_2+=reward_2
        
                next_state = env.board.copy()
                done = (win_status != 0) or env.is_draw()

                # Push to disk-based buffer for agent_2
                replay_buffer_2.push(state, action_2, reward_2, next_state, done)
                state = next_state

        # After the episode ends, train both agents
        train_agent(policy_net_1, target_net_1, optimizer_1, replay_buffer_1)
        train_agent(policy_net_2, target_net_2, optimizer_2, replay_buffer_2)

        # Periodic updates
        
        EPSILON = periodic_updates(
            episode,
            policy_net_1, target_net_1,
            policy_net_2, target_net_2,
            optimizer_1, optimizer_2,
            TRAINER_SAVE_PATH, MODEL_SAVE_PATH,
            EPSILON, EPSILON_MIN, EPSILON_DECAY,
            TARGET_UPDATE, logger
        )

        # Check final outcome for logging
        final_winner = env.check_winner()
        if final_winner == 1:
            agent_1_wins += 1
            winner_str = "Agent 1"
        elif final_winner == 2:
            agent_2_wins += 1
            winner_str = "Agent 2"
        else:
            draws += 1
            winner_str = "Draw"
        print(
            f"Episode {episode} / {NUM_EPISODES} Winner: {winner_str}. "
            f"(Agent1 wins={agent_1_wins} Total Reward:{total_reward_1}, Agent2 wins={agent_2_wins} Total Reward:{total_reward_2}, draws={draws})"
        )
        logger.info(
            f"Episode {episode} / {NUM_EPISODES} Winner: {winner_str}. "
            f"(Agent1 wins={agent_1_wins} Total Reward:{total_reward_1}, Agent2 wins={agent_2_wins} Total Reward:{total_reward_2}, draws={draws})"
        )

    logger.info("Training complete")

if __name__ == "__main__":
    main()
