import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import logging
import warnings
from numba.core.errors import NumbaPerformanceWarning
from dependencies.environment import Connect4
from dependencies.models import DQN
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import safe_make_dir
from dependencies.mcts import MCTS
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Hyperparams ----------------- #

# ---  Hardware Hyperparam --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 6
# --- Model  Hyperparam --- #
MODEL_VERSION= 35
BATCH_SIZE = 16
GAMMA = 0.90
LR = 0.00001
TARGET_EVALUATE = 100
TARGET_UPDATE = 100  
EVAL_FREQUENCY = 100
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.05
TOTAL_EPISODES = 999999999 # Infinite until it the training completed by trigerring the condition
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
def get_opponent_type(current_ep,last_ep_MCTS=0):
    global current_level_index, DYNAMIC_LEVELS
    # If we’re at the final dynamic level, use self-play, else MCTS
    if current_level_index == len(DYNAMIC_LEVELS) - 1:
        return "Self-Play"
    elif current_ep-last_ep_MCTS>10000 and last_ep_MCTS>0:
        return "None" # Complite the Train
    else:
        return "MCTS"


# ----------------- Soft Update Function ----------------- #
def soft_update(target_net, policy_net, tau=0.001):
    """
    Soft-update the target network's parameters:
        θ_target = tau * θ_policy + (1 - tau) * θ_target
    """
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

# ----------------- Training Step ----------------- 
def train_step(policy_net, target_net, optimizer, replay_buffer):
    policy_net.train()
    
    # Ensure there are enough samples in the replay buffer.
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Sample a batch from the replay buffer.
    batch = replay_buffer.sample(BATCH_SIZE)
    states       = batch["states"]           # shape: [batch, ...]
    q_actions    = batch["q_actions"]        # Action taken by the agent (from its own policy).
    rewards      = batch["rewards"]
    next_states  = batch["next_states"]
    dones        = batch["dones"]
    mcts_values  = batch["mcts_values"]      # MCTS confidence/value for its recommended action.
    mcts_actions = batch["mcts_actions"]     # Action recommended by MCTS.
    mcts_used    = batch["mcts_used"]          # Flag: True if MCTS was used (i.e. network's Q-value was below threshold).

    # Adjust shapes so that states and next_states are [batch, channels, rows, columns]
    if len(states.shape) == 5:
        states = states.squeeze(2)
    if len(states.shape) == 3:
        states = states.unsqueeze(1)
    if len(next_states.shape) == 5:
        next_states = next_states.squeeze(2)
    if len(next_states.shape) == 3:
        next_states = next_states.unsqueeze(1)

    # Compute the policy network's outputs (Q-values) for all actions.
    policy_outputs = policy_net(states)  # shape: [batch, num_actions]

    # --- Safely gather Q-values for both the agent's chosen action and the MCTS action ---
    # For transitions that represent a loss (indicated by -1 in q_actions or mcts_actions),
    # we temporarily replace -1 with 0 so that gather doesn't fail, then later reset the value.
    safe_q_actions = q_actions.clone()
    safe_mcts_actions = mcts_actions.clone()
    loss_mask_env = (q_actions != -1)
    loss_mask_mcts = (mcts_actions != -1)
    safe_q_actions[~loss_mask_env] = 0
    safe_mcts_actions[~loss_mask_mcts] = 0

    q_value_env  = policy_outputs.gather(1, safe_q_actions.unsqueeze(1)).squeeze()
    q_value_mcts = policy_outputs.gather(1, safe_mcts_actions.unsqueeze(1)).squeeze()
    
    # For invalid indices (loss transitions), set Q-values to -1.
    q_value_env[~loss_mask_env] = -1.0
    q_value_mcts[~loss_mask_mcts] = -1.0

    # Convert mcts_used to a boolean tensor.
    mcts_used = mcts_used.to(torch.bool)
    # If mcts_used is True, use the Q-value for the MCTS action; otherwise, use the agent's action.
    predicted_q = torch.where(mcts_used, q_value_mcts, q_value_env)

    # --- Compute Standard DQN Target ---
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q_values, _ = next_q_values.max(dim=1)
        dqn_targets = rewards + GAMMA * (1 - dones.float()) * max_next_q_values

    # --- Choose the Training Target ---
    # When mcts_used is True, we use mcts_values as the target; otherwise, we use dqn_targets.
    selected_target = torch.where(mcts_used, mcts_values, dqn_targets)

    # Compute the loss.
    loss = nn.MSELoss()(predicted_q, selected_target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    soft_update(target_net, policy_net, tau=0.001)

# ----------------- Checkpoint Functions ----------------- #
def load_model_checkpoint(model_path, learning_rate, buffer_size, logger, device):
    try:
        print("Starting Initialization")
        policy_net = DQN().to(device)
        logger.debug("Policy network initialized.")

        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        logger.debug("Target network initialized.")

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        logger.debug("Optimizer initialized.")

        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device,
            version=MODEL_VERSION
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
        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device,
            version=MODEL_VERSION
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
        "reset_epsilon": max(1 - 0.01 * i,0.05)
    }
    for i in range(1, 101)
]

current_level_index = 0
current_mcts_level = DYNAMIC_LEVELS[current_level_index]["mcts_simulations"]
EPSILON = DYNAMIC_LEVELS[current_level_index]["reset_epsilon"]
episodes_in_current_level = 0
recent_win_results = []
# ----------------- Dynamic Training Helpers ----------------- #
def compute_win_rate(recent_results):
    if not recent_results:
        return 0.0
    return sum(recent_results) / len(recent_results)

def update_dynamic_level(win_rate, logger):
    global current_level_index, current_mcts_level, EPSILON, episodes_in_current_level,recent_win_results
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
            recent_win_results.clear()
        else:
            episodes_in_current_level=2001
            logger.info("Maximum dynamic level reached.")

# ----------------- Simulation Function ----------------- #
def simulate_episode(args):
    """
    Simulates a single Connect4 episode using both opponent self-train logic and dynamic MCTS.
    Player 1 is controlled by an opponent whose behavior (MCTS or Self-Train) is determined
    by get_opponent_type(ep, last_mcts_ep), while player 2 is controlled by the model (agent).

    Rewards are tracked only for player 2 (total_reward) and extra statistics (mcts_count,
    low_q_value_count) are recorded.

    When player 1 wins (winner==1), the turn count is adjusted (decremented by one) so that
    total_reward truly reflects only player 2’s moves.

    Returns:
        transitions: list of (state, action, reward, next_state, done_flag, mcts_value) tuples.
        ep: episode number.
        winner: winning player (or -1 for draw).
        total_reward: cumulative reward from player 2 moves.
        turn: number of turns (adjusted when player 1 wins).
        mcts_count: count of moves where MCTS was used.
    """
    ep, current_epsilon, current_mcts_level, policy_state, logger = args
    
    
    local_policy_net = DQN().to(DEVICE)
    local_policy_net.load_state_dict(policy_state)
    local_policy_net.eval()
    
    agent = AgentLogic(local_policy_net, DEVICE)
    env = Connect4()
    state = env.reset()
    transitions = []
    done = False
    total_reward = 0.0
    winner = None
    turn = 0
    mcts_count = 0

    # Variables for self-train evaluation
    evaluate_loaded = False
    evaluator = None

    def get_opponent_action(env, logger, debug=True):
        """
        Determines the action for player 1 based on the opponent type.
        For Self-Train opponents, an evaluator may be loaded on the fly.
        Returns a triple: (action, mcts_value, mcts_used).
        """
        nonlocal evaluate_loaded, evaluator
        mcts_used=False
        last_mcts_ep = 0
        # Optionally pass the last MCTS episode if enough episodes have been played in this level.
        if episodes_in_current_level > 2000:
            last_mcts_ep = ep
        opponent_type = get_opponent_type(ep, last_mcts_ep)
        mcts_value = 0.0

        if opponent_type == "MCTS":
            mcts_used=True
            # Use dynamic MCTS level from DYNAMIC_LEVELS
            sims = current_mcts_level if current_mcts_level is not None else 0
            if sims == 0:
                mcts_action = random.choice(env.get_valid_actions())
                mcts_used=False
            else:
                mcts_agent = MCTS(logger, num_simulations=sims, debug=True)
                mcts_action, mcts_value = mcts_agent.select_action(env, env.current_player)
            if debug:
                logger.debug(f"Phase: {opponent_type}, MCTS Action SELECT={mcts_action}, MCTS_value={mcts_value}")
            return -1,mcts_action, mcts_value,mcts_used

        elif opponent_type == "Self-Train":
            if not evaluate_loaded:
                evaluator = AgentLogic(local_policy_net, device=DEVICE, q_threshold=0.8)
                torch.save(local_policy_net.state_dict(), EVAL_MODEL_PATH)
                logger.info(f"Copied local_policy_net into evaluator for evaluation from ep:{ep}.")
                evaluate_loaded = True
            # Use evaluator every TARGET_EVALUATE episodes; otherwise use agent.
            if ep % TARGET_EVALUATE == 0:
                q_action,mcts_action, mcts_value,mcts_used = evaluator.pick_action(env, current_epsilon, logger, debug=True)
                if mcts_used:
                    action=mcts_action
                else:
                    action=q_action
                if debug:
                    logger.debug(f"Phase: {opponent_type}, Evaluator Action SELECT={action}")
                return q_action,mcts_action, mcts_value,mcts_used
            else:
                q_action,mcts_action, mcts_value,mcts_used = agent.pick_action(env, current_epsilon, logger, debug=True)
                if mcts_used:
                    action=mcts_action
                else:
                    action=q_action
                if debug:
                    logger.debug(f"Phase: {opponent_type}, Self-Play Action SELECT={action}")
                return q_action,mcts_action, mcts_value,mcts_used
        else:
            # When training is completed or an unexpected type is encountered.
            return None, None,None,None
        
    # Connect 4 game
    while not done:
        turn += 1
        mcts_used = False

        if env.current_player == 1:
            # Player 1's turn (opponent)
            q_action,mcts_action, mcts_value,mcts_used= get_opponent_action(env, logger, debug=True)
            # Terminate the game if training is completed.
            if q_action is None and mcts_value is None and mcts_used is None:
                return None, None, None, None, None, None, None
            
            if mcts_used:
                action=mcts_action
            else:
                action=q_action
            env.make_move(action)
            reward_1, status = agent.compute_reward(env, action, 1)
            next_state = env.get_board().copy()
            if (status != 0) or env.is_draw(): # when 1 wins
                done = True
                transitions.append((state, action, reward_1, next_state, (status != 0 or env.is_draw()), mcts_value,mcts_action,mcts_used)) # push the reward_1
                state = next_state
                winner = status if status != 0 else -1
                if winner == 1:
                    # Compute additional reward when losing
                    reward_2, _ = agent.compute_reward(env, -1, 2)
                    #print(f"Agent lost and the final reward is: {reward_2}")
                    next_state = env.get_board().copy()
                    # add transaction where P2 lost all -1 as N/A
                    transitions.append((state, -1, reward_2, next_state, (status != 0 or env.is_draw()), -1,-1,False))
                    state = next_state
                    total_reward += reward_2
                    break
            else: # game keeps going 
                transitions.append((state, action, reward_1, next_state, (status != 0 or env.is_draw()), mcts_value,mcts_action,mcts_used))
                state = next_state
        else:  # Player2's turn (always model)
            local_policy_net.eval()
            q_action,mcts_action, mcts_value,mcts_used= agent.pick_action(env, current_epsilon,logger, debug=DEBUGMODE)
            if mcts_used:
                mcts_count+=1
                action=mcts_action
            else:
                action=q_action

            env.make_move(action)
            reward_2, status = agent.compute_reward(env, action, 2)
            total_reward += reward_2
            next_state = env.get_board().copy()
            if (status != 0) or env.is_draw():
                done = True
                transitions.append((state, action, reward_2, next_state, (status != 0 or env.is_draw()), mcts_value,mcts_action,mcts_used))
                state = next_state
                winner = status if status != 0 else -1
                break
            else:
                transitions.append((state, action, reward_2, next_state, (status != 0 or env.is_draw()), mcts_value,mcts_action,mcts_used))
                state = next_state

    return transitions, ep, winner, total_reward, turn, mcts_count


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
        device=DEVICE,
        version=MODEL_VERSION
    )

    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    global EPSILON, current_mcts_level, current_level_index, episodes_in_current_level,recent_win_results
    recent_win_results = []
    
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

    for episode in range(start_ep + 1, TOTAL_EPISODES + 1):
        episodes_in_current_level += 1

        # Parallel episode simulation
        async_result = pool.apply_async(
            simulate_episode,
            args=((episode, EPSILON, current_mcts_level, policy_net.state_dict(), logger),)
        )
        transitions, ep, winner, total_reward, turn, mcts_count = async_result.get(timeout=30)

        if transitions is None and ep is None and winner is None:
            pool.close()
            pool.join()
            logger.info("Training Completed.")
            print("Training Completed.")
        # Store transitions in replay buffer
        for transition in transitions:
            replay_buffer.push(*transition)


        mcts_rate=mcts_count/int(turn/2)
        # Update the DQN
        train_step(policy_net, target_net, optimizer, replay_buffer)

        # Track if agent (Player 2) won
        if winner == 2:
            agent_win = 1
        elif winner == -1:
            agent_win = 0.5
        else:
            agent_win = 0

        recent_win_results.append(agent_win)
        if len(recent_win_results) > WIN_RATE_WINDOW:
            recent_win_results.pop(0)
        current_win_rate = compute_win_rate(recent_win_results)

      
        logger.info(f"Episode {ep}: Winner={winner},Win Rate={current_win_rate*100:.2f}%, Turn={turn}, Reward={total_reward:.2f}, "
                        f"EPSILON={EPSILON:.6f}, MCTS LEVEL={current_mcts_level}, "
                        f"MCTS used Rate:{mcts_rate*100:.2f}%")
        print(f"Episode {ep}: Winner={winner},Win Rate={current_win_rate*100:.2f}%, Turn={turn}, Reward={total_reward:.2f}, "
                        f"EPSILON={EPSILON:.6f}, MCTS LEVEL={current_mcts_level}, "
                        f"MCTS used Rate:{mcts_rate*100:.2f}%")

        # Possibly advance dynamic training level
        update_dynamic_level(current_win_rate, logger)

        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        # Periodic saving + target network update
        periodic_updates(episode, policy_net, target_net, optimizer, logger)

    pool.close()
    pool.join()
    logger.info("Training Cmpleted.")
    print("Training Completed.")
if __name__ == "__main__":
    mp.freeze_support()
    run_training()
