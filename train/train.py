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
import time

# ----------------- Hyperparams ----------------- #
BATCH_SIZE = 128
GAMMA = 0.95
LR = 0.001

EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.05
TARGET_EVALUATE = 100
BACKUP_FREQUENCY=100
TARGET_UPDATE = 1
TOTAL_EPISODES = 1100000
REPLAY_BUFFER_SIZE = TOTAL_EPISODES
REPLAY_CAPACITY = TOTAL_EPISODES
RAND_EPISODE_BY = 0
MCTS_EPISODE_BY = 1000000
DEBUGMODE = True
EVAL_FREQUENCY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'Connect4_Agent_Model.pth'
BACKUP_PATH = 'BACKUP.pth'
EVAL_MODEL_PATH = 'Connect4_Agent_EVAL.pth'
NUM_WORKERS = 6  # Adjust as needed
MAX_MCTS=2000
# ----------------- Utility Function ----------------- #
def get_opponent_type(ep):
    """
    Returns the type of opponent based on the current episode number.
    """
    if ep < RAND_EPISODE_BY:
        return "Random"
    elif ep < MCTS_EPISODE_BY:
        return "MCTS"
    else:
        return "Self-Play"

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
    logger.debug(f"LOSS:{loss}")
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
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        logger.debug("Optimizer initialized.")
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.debug("Replay buffer initialized.")
        start_episode = 0
        if os.path.exists(model_path):
            print("Loading the previous Data")
            checkpoint = torch.load(model_path, map_location=device)
            logger.debug(f"Checkpoint file path: {model_path} verified.")
            if 'policy_net_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                logger.debug("Loaded policy network state_dict.")
            else:
                policy_net.load_state_dict(checkpoint)
                logger.debug("Loaded raw policy network state_dict.")
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.debug("Loaded optimizer state_dict.")
            start_episode = checkpoint.get('episode', 0)
            logger.debug(f"Loaded start episode: {start_episode}")
        else:
            logger.warning(f"Checkpoint file {model_path} does not exist. Starting fresh.")
    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}. Recovering using backup {BACKUP_PATH}.")
        print(f"Failed to load model from {model_path}: {e}. Recovering using backup {BACKUP_PATH}.")
        try:
            if os.path.exists(BACKUP_PATH):
                checkpoint = torch.load(BACKUP_PATH, map_location=device)
                logger.debug(f"Checkpoint file path: {BACKUP_PATH} verified.")
                if 'policy_net_state_dict' in checkpoint:
                    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    logger.debug("Loaded policy network state_dict.")
                else:
                    policy_net.load_state_dict(checkpoint)
                    logger.debug("Loaded raw policy network state_dict.")
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.debug("Loaded optimizer state_dict.")
                start_episode = checkpoint.get('episode', 0)
                logger.debug(f"Copying the Data")
                save_model_checkpoint(model_path, policy_net, target_net, optimizer, start_episode, logger) # Copy the data
                logger.debug(f"Loaded start episode: {start_episode}")
        except Exception as e:
            print(f"Failed to load model from {BACKUP_PATH}: {e}. Starting fresh.")
            logger.critical(f"Failed to load model from {BACKUP_PATH}: {e}. Starting fresh.")
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
        logger.debug(f"Saved model checkpoint to {model_path} at episode {episode}.")
    except Exception as e:
        logger.error(f"Failed to save model checkpoint to {model_path}: {e}.")

def periodic_updates(episode, policy_net, target_net, optimizer,logger):
    global EPSILON
    try:
        if episode % TARGET_UPDATE == 0:
            save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, episode, logger)
            logger.debug(f"Models saved at episode {episode}")
            target_net.load_state_dict(policy_net.state_dict())
            logger.debug("Target network updated")
        if episode % BACKUP_FREQUENCY== 0:
            save_model_checkpoint(BACKUP_PATH, policy_net, target_net, optimizer, episode, logger)
            logger.debug(f"Backup saved at episode {episode}")
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON

# ----------------- Multiprocessing Simulation Function (GPU-based) ----------------- #
def simulate_episode(args):
    """
    Simulate a single game (episode) and return its transitions, the episode number,
    the winner, the total reward, and the number of turns (game length).
    Each process loads its own copy of the model onto the GPU.
    """
    ep, current_epsilon, policy_state = args
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
    turn = 0
    wins = 0
    losses = 0
    draws = 0
    evaluate_loaded = False
    evaluator = None
    logger = logging.getLogger("SimulateEpisode")
    
    while not done:
        turn += 1
        mcts_used = False
        if env.current_player == 1:
            # Determine opponent type
            opponent_type = get_opponent_type(ep)

            def get_opponent_action(env, debug=True):
                nonlocal evaluate_loaded, evaluator
                if opponent_type == "Random":
                    action = random.choice(env.get_valid_actions())
                    if debug:
                        logger.debug(f"Phase: {opponent_type}, Random Action SELECT={action}")
                    return action
                elif opponent_type == "MCTS":
                    scaling_factor = MAX_MCTS / (MCTS_EPISODE_BY - RAND_EPISODE_BY)
                    mcts_level = ep - RAND_EPISODE_BY
                    sims = int(scaling_factor * mcts_level)
                    sims = min(MAX_MCTS, sims)
                    if sims == 0:
                        action = random.choice(env.get_valid_actions())
                    else:  
                        mcts_action = MCTS(num_simulations=sims, debug=True)
                        action = mcts_action.select_action(env, env.current_player)
                    if debug:
                        logger.debug(f"Phase: {opponent_type}, MCTS Action SELECT={action}")
                    return action
                elif opponent_type == "Self-Play":
                    if not evaluate_loaded:
                        evaluator = AgentLogic(local_policy_net, device=device, q_threshold=0.5)
                        torch.save(local_policy_net.state_dict(), EVAL_MODEL_PATH)
                        logger.info(f"Copied local_policy_net into evaluator for evaluation from ep:{ep}.")
                        evaluate_loaded = True
                    if ep % TARGET_EVALUATE == 0:
                        action = evaluator.pick_action(env,  current_epsilon, episode=ep, debug=DEBUGMODE)
                        logger.debug(f"Phase: {opponent_type}, Evaluator Action SELECT={action}")
                        return action, mcts_used
                    else:
                        action = agent.pick_action(env,  current_epsilon, episode=ep, debug=DEBUGMODE)
                        logger.debug(f"Phase: {opponent_type}, Self-Play Action SELECT={action}")
                        return action, mcts_used

            # Player1's turn
            action = get_opponent_action(env, debug=DEBUGMODE)
            env.make_move(action)
            reward, status = agent.compute_reward(env, action, 1,mcts_used)
            next_state = env.get_board().copy()
            if (status != 0) or env.is_draw():
                done = True
                transitions.append((state, action, reward, next_state, True))
                state = next_state
                winner = status if status != 0 else -1
                if winner == 2:
                    wins += 1
                elif winner == 1:
                    # Compute additional reward when losing
                    reward, _ = agent.compute_reward(env, -1, 2,mcts_used)
                    next_state = env.get_board().copy()
                    transitions.append((state, action, reward, next_state, True))
                    state = next_state
                    total_reward += reward
                    losses += 1
                elif winner == -1:
                    draws += 1
                break
            else:
                transitions.append((state, action, reward, next_state, False))
                state = next_state
                total_reward += reward
        else:  # Player2's turn (always model)
            local_policy_net.eval()
            action,mcts_used = agent.pick_action(env, current_epsilon, episode=ep, debug=DEBUGMODE)
            env.make_move(action)
            reward, status = agent.compute_reward(env, action, 2,mcts_used)
            total_reward += reward
            next_state = env.get_board().copy()
            if (status != 0) or env.is_draw():
                done = True
                transitions.append((state, action, reward, next_state, True))
                state = next_state
                winner = status if status != 0 else -1
                break
            else:
                transitions.append((state, action, reward, next_state, False))
                state = next_state

    return transitions, ep, winner, total_reward, turn

# ----------------- Main Function with Multiprocessing and Timeout ----------------- #
def run_training():
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Connect4Logger")
    
    
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
        MODEL_SAVE_PATH, LR, REPLAY_BUFFER_SIZE, logger, DEVICE
    )
    
    global EPSILON
    EPSILON = max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** start_ep))
    logger.debug(f"Continue from episode {start_ep}. Current EPSILON: {EPSILON:.3f}")
    print(f"Continue from episode {start_ep}. Current EPSILON: {EPSILON:.3f}")
    
    total_episodes = TOTAL_EPISODES
    pool = mp.Pool(processes=NUM_WORKERS)
    
    for batch_start in range(start_ep + 1, total_episodes + 1, NUM_WORKERS):
        async_results = []
        for ep in range(batch_start, min(batch_start + NUM_WORKERS, total_episodes + 1)):
            async_results.append(
                pool.apply_async(simulate_episode, args=((ep, EPSILON, policy_net.state_dict()),))
            )
        
        for async_res in async_results:
            try:
                transitions, ep, winner, total_reward, turn = async_res.get(timeout=15)
            except mp.TimeoutError:
                logger.error("A simulation episode timed out and will be skipped.")
                continue
            for transition in transitions:
                replay_buffer.push(*transition)
            train_step(policy_net, target_net, optimizer, replay_buffer, logger)
            if ep>RAND_EPISODE_BY:
                sims=int( MAX_MCTS /(MCTS_EPISODE_BY-RAND_EPISODE_BY)*ep)
            logger.info(f"Episode {ep}: Winner={winner}, Turn={turn}, Reward={total_reward:.2f}, EPSILON={EPSILON:.6f}, Simulation={sims}")
            print(f"Episode {ep}: Winner={winner}, Turn={turn}, Reward={total_reward:.2f}, EPSILON={EPSILON:.6f}, Simulation={sims}")
            EPSILON = periodic_updates(ep, policy_net, target_net, optimizer,logger)
    
    pool.close()
    pool.join()
    save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, total_episodes, logger)
    logger.info("Training finished.")

def main():
    max_restarts = 5
    restart_count = 0
    while restart_count < max_restarts:
        try:
            run_training()
            break
        except TimeoutError as e:
            torch.cuda.empty_cache()
            print(f"Restarting training loop due to timeout: {e}")
            restart_count += 1
            time.sleep(3)
    if restart_count >= max_restarts:
        print("Exceeded maximum number of restarts. Exiting.")

if __name__ == "__main__":
    mp.freeze_support()  # For Windows support
    mp.set_start_method('spawn', force=True)
    main()
