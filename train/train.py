import os
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp
import logging
from enviroment import Connect4
from models import DQN
from agent import AgentLogic
from replay_buffer import DiskReplayBuffer
from utils import setup_logger, safe_make_dir, get_next_index
from mcts import MCTS

# ----------------- Hyperparams ----------------- #
BATCH_SIZE = 128
GAMMA = 0.95
LR = 0.001
REPLAY_CAPACITY = 100000
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.05
REPLAY_BUFFER_SIZE = 10000
TARGET_EVALUATE = 100
TARGET_UPDATE = 500
TOTAL_EPISODES = 2500000
RAND_EPISODE_BY = 1000000
MCTS_EPISODE_BY = 2000000
DEBUGMODE = False
EVAL_FREQUENCY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'Connect4_Agent_Model.pth'
EVAL_MODEL_PATH = 'Connect4_Agent_EVAL.pth'
NUM_WORKERS=10

# ----------------- Training Step ----------------- #
def train_step(policy_net, target_net, optimizer, replay_buffer, logger):
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = (
        batch["states"],
        batch["actions"],
        batch["rewards"],
        batch["next_states"],
        batch["dones"],
    )

    # Ensure proper shape
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
        next_q = target_net(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones.float()) * GAMMA * next_q

    loss = nn.MSELoss()(q_values, targets)
    logger.info(f"LOSS:{loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----------------- Checkpoint Functions ----------------- #
def load_model_checkpoint(model_path, learning_rate, buffer_size, logger, device):
    try:
        policy_net = DQN(device=device).to(device)
        logger.info("Policy network initialized.")
        target_net = DQN(device=device).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        logger.info("Target network initialized.")
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        logger.info("Optimizer initialized.")
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.info("Replay buffer initialized.")
        start_episode = 0
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"Checkpoint file path: {model_path} verified.")
            if 'policy_net_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                logger.info("Loaded policy network state_dict.")
            else:
                policy_net.load_state_dict(checkpoint)
                logger.info("Loaded raw policy network state_dict.")
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state_dict.")
            start_episode = checkpoint.get('episode', 0)
            logger.info(f"Loaded start episode: {start_episode}")
        else:
            logger.warning(f"Checkpoint file {model_path} does not exist. Starting fresh.")
    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh.")
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
        logger.info("Re-initialized components.")
        start_episode = 0
    return policy_net, target_net, optimizer, replay_buffer, start_episode

def save_model_checkpoint(model_path, policy_net, target_net, optimizer, episode, logger):
    try:
        checkpoint = {
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved model checkpoint to {model_path} at episode {episode}.")
    except Exception as e:
        logger.error(f"Failed to save model checkpoint to {model_path}: {e}.")

def periodic_updates(episode, policy_net, target_net, optimizer, MODEL_SAVE_PATH, EPSILON, logger):
    try:
        if episode % TARGET_UPDATE == 0:
            save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, episode, logger)
            logger.info(f"Models saved at episode {episode}")
            target_net.load_state_dict(policy_net.state_dict())
            logger.info("Target network updated")
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON

# ----------------- Parallel Simulation Function ----------------- #
def simulate_episode(args):
    """
    This function simulates a single game (episode) and returns the transitions,
    the episode number, the winner, and the total reward.
    """
    ep, current_epsilon, policy_state = args
    # Create a local copy of the policy network for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_policy_net = DQN(device=device).to(device)
    local_policy_net.load_state_dict(policy_state)
    local_policy_net.eval()
    agent = AgentLogic(local_policy_net, device=device, q_threshold=0.8)
    env = Connect4()
    state = env.reset()
    transitions = []
    done = False
    total_reward = 0.0
    winner = None

    # For demonstration, we alternate turns.
    while not done:
        if env.current_player == 1:
            # For simplicity in this example, use random moves for player 1.
            action = random.choice(env.get_valid_actions())
        else:
            # Player 2 uses the agent
            action = agent.pick_action(env, env.current_player, current_epsilon, episode=ep, debug=False)
        env.make_move(action)
        reward, status = agent.compute_reward(env, action, env.current_player)
        next_state = env.get_board().copy()
        # If the game ends, mark done and set winner accordingly
        if status != 0 or env.is_draw():
            done = True
            winner = status if status != 0 else -1
        transitions.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

    return transitions, ep, winner, total_reward

# ----------------- Main Function with Multiprocessing ----------------- #
def main():
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Connect4Logger")
    print("Starting Initialization")

    replay_buffer = DiskReplayBuffer(
        capacity=REPLAY_CAPACITY,
        state_shape=(6, 7),
        device=DEVICE
    )

    policy_net = DQN(device=DEVICE).to(DEVICE)
    target_net = DQN(device=DEVICE).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    policy_net, target_net, optimizer, replay_buffer, start_ep = load_model_checkpoint(
        MODEL_SAVE_PATH,
        LR,
        REPLAY_BUFFER_SIZE,
        logger,
        DEVICE
    )

    # Adjust epsilon based on the starting episode
    global EPSILON
    EPSILON = max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** start_ep))
    logger.info(f"Continue from episode {start_ep}. Current EPSILON: {EPSILON:.3f}")
    print(f"Continue from episode {start_ep}. Current EPSILON: {EPSILON:.3f}")

    num_workers = NUM_WORKERS  # Set the number of parallel games to run
    pool = mp.Pool(processes=num_workers)
    total_episodes = TOTAL_EPISODES

    # Run episodes in batches concurrently
    for batch_start in range(start_ep + 1, total_episodes + 1, num_workers):
        batch_args = []
        for ep in range(batch_start, min(batch_start + num_workers, total_episodes + 1)):
            # Pass the current episode number, current epsilon, and current policy parameters
            batch_args.append((ep, EPSILON, policy_net.state_dict()))
        results = pool.map(simulate_episode, batch_args)

        for transitions, ep, winner, total_reward in results:
            for transition in transitions:
                replay_buffer.push(*transition)
            train_step(policy_net, target_net, optimizer, replay_buffer, logger)
            logger.info(f"Episode {ep}: Winner={winner}, Total Reward={total_reward:.2f}, EPSILON={EPSILON:.3f}")
            print(f"Episode {ep}: Winner={winner}, Total Reward={total_reward:.2f}, EPSILON={EPSILON:.3f}")
            # Periodic updates (checkpoint saving, target network update, epsilon decay)
            EPSILON = periodic_updates(ep, policy_net, target_net, optimizer, MODEL_SAVE_PATH, EPSILON, logger)

    pool.close()
    pool.join()
    save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, total_episodes, logger)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
