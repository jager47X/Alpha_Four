import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp
import logging
import warnings
from numba.core.errors import NumbaPerformanceWarning
from dependencies.environment import Connect4
from dependencies.models import DQN
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import setup_logger, safe_make_dir, get_next_index
from dependencies.mcts import MCTS

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Hyperparams ----------------- #

# ---  Hardware Hyperparam --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 6
# --- Model  Hyperparam --- #
MODEL_VERSION= 4
BATCH_SIZE = 128
GAMMA = 0.95
LR = 0.0001
TARGET_EVALUATE = 100
TARGET_UPDATE = 100  
EVAL_FREQUENCY = 100
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05
SELF_PLAY = 100000
DEBUGMODE = True
# --- MCTS  Hyperparam --- #
WIN_RATE_WINDOW = 100
MAX_MCTS = 2000
MIN_EPISODES_PER_LEVEL = 100  # Minimum episodes at each dynamic level before advancing
WIN_RATE_THRESHOLD=0.80 # Minimum win rate to pass the mcts level
# ---  file paths under dat/ directory --- #
MODEL_SAVE_PATH = f'./data/models/{MODEL_VERSION}/Connect4_Agent_Model.pth'
EVAL_MODEL_PATH = f'./data/models/{MODEL_VERSION}/Connect4_Agent_EVAL.pth'
LOGS_DIR=f'./data/logs/train_logs/{MODEL_VERSION}'
LOG_FILE=f'./data/logs/train_logs/{MODEL_VERSION}/train.log'
# ----------------- Opponent Type Function ----------------- #
def get_opponent_type(ep):
    global current_level_index, DYNAMIC_LEVELS
    # If we’re at the final dynamic level, use self-play, else MCTS
    if current_level_index == len(DYNAMIC_LEVELS) - 1:
        return "Self-Play"
    else:
        return "MCTS"

# ----------------- Training Step ----------------- #
def train_step(policy_net, target_net, optimizer, replay_buffer, logger):
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones, mcts_values = (
        batch["states"],
        batch["actions"],
        batch["rewards"],
        batch["next_states"],
        batch["dones"],
        batch["mcts_values"],
    )

    # Adjust shapes so [batch, channels, rows, columns]
    if len(states.shape) == 5:
        states = states.squeeze(2)
    if len(states.shape) == 3:
        states = states.unsqueeze(1)
    if len(next_states.shape) == 5:
        next_states = next_states.squeeze(2)
    if len(next_states.shape) == 3:
        next_states = next_states.unsqueeze(1)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q_values, _ = next_q_values.max(dim=1)
        dqn_targets = rewards + GAMMA * (1 - dones.float()) * max_next_q_values

    # Blended target: half from MCTS value, half from standard DQN
    mcts_values = mcts_values.clone().detach().float().to(states.device)
    lambda_weight = 0.5
    blended_targets = lambda_weight * mcts_values + (1 - lambda_weight) * dqn_targets

    loss = nn.MSELoss()(q_values, blended_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----------------- Checkpoint Functions ----------------- #
def load_model_checkpoint(model_path, learning_rate, buffer_size, logger, device):
    try:
        print("Starting Initialization")
        policy_net = DQN(device=device).to(device)
        logger.debug("Policy network initialized.")

        target_net = DQN(device=device).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        logger.debug("Target network initialized.")

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        logger.debug("Optimizer initialized.")

        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.debug("Replay buffer initialized.")
        start_episode = 0

        if os.path.exists(model_path):
            print("Loading previous data...")
            checkpoint = torch.load(model_path, map_location=device)
            logger.debug(f"Checkpoint file found at: {model_path}")
            if 'policy_net_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                logger.debug("Loaded policy network state_dict.")
            else:
                # Legacy checkpoint format
                policy_net.load_state_dict(checkpoint)
                logger.debug("Loaded raw policy network state_dict.")

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.debug("Loaded optimizer state_dict.")

            start_episode = checkpoint.get('episode', 0)
            global current_level_index, current_mcts_level, EPSILON, episodes_in_current_level
            current_level_index = checkpoint.get('current_level_index', 0)
            current_mcts_level = checkpoint.get('current_mcts_level',
                                               DYNAMIC_LEVELS[current_level_index]["mcts_simulations"])
            EPSILON = checkpoint.get('EPSILON',
                                     DYNAMIC_LEVELS[current_level_index]["reset_epsilon"])
            episodes_in_current_level = checkpoint.get('episodes_in_current_level', 0)
            logger.debug(
                f"Loaded start_ep={start_episode}, Level={current_level_index}, "
                f"MCTS={current_mcts_level}, EPSILON={EPSILON}, "
                f"EpisodesInLevel={episodes_in_current_level}"
            )
        else:
            logger.warning(f"Checkpoint file {model_path} does not exist. Starting fresh.")

    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh.")
        print(f"Failed to load model from {model_path}: {e}. Starting fresh.")
        policy_net = DQN(device=device).to(device)
        target_net = DQN(device=device).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.debug("Re-initialized components.")
        start_episode = 0

    return policy_net, target_net, optimizer, replay_buffer, start_episode

def save_model_checkpoint(model_path, policy_net, target_net, optimizer, episode, logger):
    try:
        checkpoint = {
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'current_level_index': current_level_index,
            'current_mcts_level': current_mcts_level,
            'EPSILON': EPSILON,
            'episodes_in_current_level': episodes_in_current_level
        }
        torch.save(checkpoint, model_path)
        logger.debug(f"Saved model checkpoint to {model_path} at episode {episode}.")
    except Exception as e:
        logger.error(f"Failed to save model checkpoint to {model_path}: {e}")

def periodic_updates(episode, policy_net, target_net, optimizer, logger):
    global EPSILON
    if episode % 100 == 0:
        try:
            save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, episode, logger)
            logger.debug(f"Models saved at episode {episode}")
            target_net.load_state_dict(policy_net.state_dict())
            logger.debug("Target network updated")
        except Exception as e:
            logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON

# ----------------- Dynamic MCTS Levels ----------------- #
DYNAMIC_LEVELS = [
    {
        "mcts_simulations": 20 * i,
        "win_rate_threshold": WIN_RATE_THRESHOLD,
        "reset_epsilon": max(1 - 0.01 * i, 0.75)
    }
    for i in range(1, 101)
]

current_level_index = 0
current_mcts_level = DYNAMIC_LEVELS[current_level_index]["mcts_simulations"]
EPSILON = DYNAMIC_LEVELS[current_level_index]["reset_epsilon"]
episodes_in_current_level = 0

# ----------------- Dynamic Training Helpers ----------------- #
def compute_win_rate(recent_results):
    if not recent_results:
        return 0.0
    return sum(recent_results) / len(recent_results)

def update_dynamic_level(win_rate, logger):
    global current_level_index, current_mcts_level, EPSILON, episodes_in_current_level
    threshold = DYNAMIC_LEVELS[current_level_index]["win_rate_threshold"]
    if episodes_in_current_level >= MIN_EPISODES_PER_LEVEL and win_rate >= threshold:
        logger.info(
            f"After {episodes_in_current_level} episodes, win rate {win_rate*100:.2f}% "
            f"reached threshold {threshold*100:.2f}%."
        )
        if current_level_index + 1 < len(DYNAMIC_LEVELS):
            current_level_index += 1
            current_mcts_level = DYNAMIC_LEVELS[current_level_index]["mcts_simulations"]
            EPSILON = DYNAMIC_LEVELS[current_level_index]["reset_epsilon"]
            logger.info(
                f"Advanced to Level {current_level_index}: MCTS={current_mcts_level}, "
                f"EPSILON reset to {EPSILON:.2f}"
            )
            episodes_in_current_level = 0
        else:
            logger.info("Maximum dynamic level reached.")

# ----------------- Simulation Function ----------------- #
def simulate_episode(args):
    ep, current_epsilon, current_mcts_level, policy_state, logger = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_policy_net = DQN(device=device).to(device)
    local_policy_net.load_state_dict(policy_state)
    local_policy_net.eval()

    agent = AgentLogic(local_policy_net, device=device, q_threshold=0.5)
    env = Connect4()
    state = env.reset()
    transitions = []
    done = False
    total_reward = 0.0
    winner = None
    turn = 0
    mcts_count = 0
    low_q_value_count = 0

    def get_opponent_action(env, logger, debug=True):
        opponent_type = get_opponent_type(ep)
        mcts_value = 0.0

        if opponent_type == "Random":
            action = random.choice(env.get_valid_actions())
            if debug:
                logger.debug(f"Phase: {opponent_type}, Random Action SELECT={action}")
            return action, mcts_value, False

        elif opponent_type == "MCTS":
            sims = current_mcts_level
            if sims == 0:
                action = random.choice(env.get_valid_actions())
            else:
                mcts_agent = MCTS(logger, num_simulations=sims, debug=True)
                action, mcts_value = mcts_agent.select_action(env, env.current_player)
            if debug:
                logger.debug(f"Phase: {opponent_type}, MCTS Action SELECT={action}, MCTS_value={mcts_value}")
            return action, mcts_value, True

        elif opponent_type == "Self-Play":
            action = agent.pick_action(env, current_epsilon, logger, debug=True)
            if debug:
                logger.debug(f"Phase: {opponent_type}, Self-Play Action SELECT={action}")
            return action, mcts_value, False

    while not done:
        turn += 1
        mcts_used = False
        low_q_value = False

        if env.current_player == 1:
            action, mcts_value, mcts_used = get_opponent_action(env, logger, debug=True)
            env.make_move(action)
            reward, status = agent.compute_reward(env, action, 1, mcts_used, current_epsilon, low_q_value)
            next_state = env.get_board().copy()
            transitions.append((state, action, reward, next_state, (status != 0 or env.is_draw()), mcts_value))
            state = next_state
            total_reward += reward
            if status != 0 or env.is_draw():
                winner = status if status != 0 else -1
                break
        else:
            action, mcts_value, mcts_used = agent.pick_action(
                env, current_epsilon, logger, debug=DEBUGMODE, return_mcts_value=True
            )
            if mcts_used:
                mcts_count += 1
            if low_q_value:
                low_q_value_count += 1

            env.make_move(action)
            reward, status = agent.compute_reward(env, action, 2, mcts_used, current_epsilon, low_q_value)
            total_reward += reward

            next_state = env.get_board().copy()
            transitions.append((state, action, reward, next_state, (status != 0 or env.is_draw()), mcts_value))
            state = next_state

            if status != 0 or env.is_draw():
                winner = status if status != 0 else -1
                break

    return transitions, ep, winner, total_reward, turn, mcts_count, low_q_value_count

# ----------------- Training Loop ----------------- #
def run_training():
    # Put training logs under ./dat/logs
    safe_make_dir(LOGS_DIR) 
    logging.basicConfig(
        filename=LOG_FILE,
        filemode='a',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Connect4Logger")

    # Ensure the directory for model saving exists
    safe_make_dir(os.path.dirname(MODEL_SAVE_PATH))

    replay_buffer = DiskReplayBuffer(
        capacity=100000,
        state_shape=(6, 7),
        device=DEVICE
    )

    policy_net = DQN(device=DEVICE).to(DEVICE)
    target_net = DQN(device=DEVICE).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    global EPSILON, current_mcts_level, current_level_index, episodes_in_current_level
    recent_win_results = []
    total_episodes = SELF_PLAY
    pool = mp.Pool(processes=NUM_WORKERS)

    # Load checkpoint from disk.
    policy_net, target_net, optimizer, replay_buffer, start_ep = load_model_checkpoint(
        MODEL_SAVE_PATH, LR, 100000, logger, DEVICE
    )

    # If it's a fresh start, use default epsilon; else continue from checkpoint
    if start_ep == 0:
        EPSILON = DYNAMIC_LEVELS[current_level_index]["reset_epsilon"]
    else:
        EPSILON = max(EPSILON_MIN, EPSILON)

    logger.debug(f"Continue from episode {start_ep}. EPSILON: {EPSILON:.3f}, MCTS: {current_mcts_level}")
    print(f"Continue from episode {start_ep}. EPSILON: {EPSILON:.3f}, MCTS: {current_mcts_level}")

    for episode in range(start_ep + 1, total_episodes + 1):
        episodes_in_current_level += 1

        # Parallel episode simulation
        async_result = pool.apply_async(
            simulate_episode,
            args=((episode, EPSILON, current_mcts_level, policy_net.state_dict(), logger),)
        )
        transitions, ep, winner, total_reward, turn, mcts_count, low_q_value_count = async_result.get(timeout=15)

        # Store transitions in replay buffer
        for transition in transitions:
            replay_buffer.push(*transition)

        # Update the DQN
        train_step(policy_net, target_net, optimizer, replay_buffer, logger)

        # Track if agent (Player 2) won
        agent_win = 1 if winner == 2 else 0
        recent_win_results.append(agent_win)
        if len(recent_win_results) > WIN_RATE_WINDOW:
            recent_win_results.pop(0)
        current_win_rate = compute_win_rate(recent_win_results)

      
        logger.info(f"Episode {ep}: Winner={winner},Win Rate={current_win_rate*100:.2f}%, Turn={turn}, Reward={total_reward:.2f}, "
                        f"EPSILON={EPSILON:.6f}, MCTS LEVEL={current_mcts_level}, "
                        f"Cumulative MCTS used: {mcts_count}/{int(turn/2)}, Recalculation: {low_q_value_count}")
        print(
            f"Episode {ep}: Win Rate={current_win_rate*100:.2f}%, "
            f"MCTS={current_mcts_level}, EPSILON={EPSILON:.6f}"
        )

        # Possibly advance dynamic training level
        update_dynamic_level(current_win_rate, logger)

        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        # Periodic saving + target network update
        periodic_updates(episode, policy_net, target_net, optimizer, logger)

    pool.close()
    pool.join()
    logger.info("Training finished.")

if __name__ == "__main__":
    mp.freeze_support()
    run_training()
