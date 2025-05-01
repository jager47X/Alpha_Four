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
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import safe_make_dir
from dependencies.mcts import MCTS
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Hyperparams ----------------- #

# ---  Hardware Hyperparam --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 6
REPLAYBUFFER_CAPACITY=10000000
# --- Model  Hyperparam --- #
MODEL_VERSION= 63
BATCH_SIZE = 64
GAMMA = 0.99 
LR = 0.0005
TARGET_EVALUATE = 100
TARGET_UPDATE = 100  
EVAL_FREQUENCY = 100
EPSILON_DECAY = 0.99999 
EPSILON_MIN = 0.05
TOTAL_EPISODES = 999999999 # Infinite until it the training completed by trigerring the condition
DEBUGMODE = True
TAU=0.001
Q_THRESHOLD=0.6
HYBRID_THRESHOLD =0.5
TEMPERATURE=1.0
TAU = 0.01
DEBUGMODE = True
ALPHA_DISTILL = 0.1  # Weight for the policy distillation loss
DISTILL_TEMPERATURE = 1.0  # Temperature for softmax on DQN Q-values
if MODEL_VERSION>=45:
   from dependencies.layer_models.model2 import DQN
else:
   from dependencies.layer_models.model1 import DQN
# --- MCTS  Hyperparam --- #
WIN_RATE_WINDOW = 100
MCTS_SIMULATIONS=2000
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

# ----------------- train_step Function ----------------- #

def train_step(policy_net, target_net, optimizer, replay_buffer):
    """
    Perform one training step on a batch of transitions.

    Now includes *policy distillation* from MCTS:
      - If 'mcts_policy' is in the replay buffer, we
        do cross-entropy between DQN's softmax distribution
        and MCTS's distribution for that transition.
    """

    policy_net.train()

    if len(replay_buffer) < BATCH_SIZE:
        return

    # --- 1) Sample a batch from the replay buffer ---
    batch = replay_buffer.sample(BATCH_SIZE)

    states        = batch["states"]         # shape: [B, ...]
    actions       = batch["q_actions"]      # int action (DQN's action or -1 if invalid)
    rewards       = batch["rewards"]
    next_states   = batch["next_states"]
    dones         = batch["dones"]

    # MCTS info
    mcts_values   = batch.get("mcts_values",   None)  # in [0,1]
    hybrid_values = batch.get("hybrid_values", None)
    mcts_actions  = batch.get("mcts_actions",  None)
    model_used    = batch.get("model_used",    None)  # 0=dqn,1=mcts,2=hybrid,3=dqn

    # The new distribution we want to distill from, shape [B, num_actions], each row sums to 1
    mcts_policy_dist = batch.get("mcts_policy", None)

    # --- 2) Fix shapes for CNN input ---
    if states.ndimension() == 5:
        states = states.squeeze(2)
    elif states.ndimension() == 3:
        states = states.unsqueeze(1)

    if next_states.ndimension() == 5:
        next_states = next_states.squeeze(2)
    elif next_states.ndimension() == 3:
        next_states = next_states.unsqueeze(1)

    # --- 3) Compute all Q-values from current policy_net ---
    policy_outputs = policy_net(states)  # shape: [B, num_actions]

    # (optional) debug info
    if DEBUGMODE:
        print("=== DEBUG: Shapes ===")
        print(f"states.shape: {states.shape}, next_states.shape: {next_states.shape}")
        print(f"policy_outputs[0]: {policy_outputs[0].detach().cpu().numpy()}")

    # --- 4) Gather Q-values for the *action actually taken* ---
    # Here we still separate 'env' action from 'mcts_action', as in your code
    q_actions_list = [a if a != -1 else 0 for a in actions]
    mcts_actions_list = [a if a != -1 else 0 for a in mcts_actions]

    loss_mask_env  = torch.tensor([a != -1 for a in actions], device=states.device)
    loss_mask_mcts = torch.tensor([a != -1 for a in mcts_actions], device=states.device)

    safe_q_actions    = torch.tensor(q_actions_list,    device=states.device, dtype=torch.int64)
    safe_mcts_actions = torch.tensor(mcts_actions_list, device=states.device, dtype=torch.int64)

    q_value_env  = policy_outputs.gather(1, safe_q_actions.unsqueeze(1)).squeeze(1)
    q_value_mcts = policy_outputs.gather(1, safe_mcts_actions.unsqueeze(1)).squeeze(1)

    # Force invalid actions to -1
    q_value_env[~loss_mask_env]   = -1.0
    q_value_mcts[~loss_mask_mcts] = -1.0

    # --- 5) Build the predicted Q-value depending on who chose the action ---
    mode_dict = {0: "dqn", 1: "mcts", 2: "hybrid", 3: "dqn"}
    predicted_q_list = []
    B = policy_outputs.shape[0]

    for i in range(B):
        if model_used is not None:
            mode_code = model_used[i].item() if hasattr(model_used[i], "item") else model_used[i]
            mode_str  = mode_dict.get(mode_code, "dqn")
        else:
            mode_str  = "dqn"

        if mode_str == "mcts":
            predicted_q_list.append(q_value_mcts[i])
        elif mode_str == "hybrid":
            blended_q = 0.5 * (q_value_env[i] + q_value_mcts[i])
            predicted_q_list.append(blended_q)
        else:  # dqn
            predicted_q_list.append(q_value_env[i])

    predicted_q = torch.stack(predicted_q_list)

    # --- 6) Compute standard DQN target ---
    with torch.no_grad():
        next_q_vals = target_net(next_states)
        max_next_q_vals, _ = next_q_vals.max(dim=1)
        dqn_targets = rewards + GAMMA * (1 - dones.float()) * max_next_q_vals

    # Scale MCTS from [0,1] -> [-100, 100] if needed
    if mcts_values is not None:
        scaled_mcts_values = mcts_values * 200.0 - 100.0
    else:
        scaled_mcts_values = torch.zeros_like(dqn_targets)

    # Build final target per sample
    target_list = []
    for i in range(B):
        if model_used is not None:
            mode_code = model_used[i].item() if hasattr(model_used[i], "item") else model_used[i]
            mode_str  = mode_dict.get(mode_code, "dqn")
        else:
            mode_str  = "dqn"

        if mode_str == "mcts":
            target_list.append(scaled_mcts_values[i])
        elif mode_str == "hybrid":
            if hybrid_values is not None:
                target_list.append(hybrid_values[i])
            else:
                # fallback
                target_list.append(dqn_targets[i])
        else:  # "dqn"
            target_list.append(dqn_targets[i])

    selected_target = torch.stack(target_list)

    # Mask out invalid transitions
    loss_mask = torch.tensor([a != -1 for a in actions], device=states.device)
    predicted_q_final = torch.where(loss_mask, predicted_q, torch.full_like(predicted_q, -1.0))
    selected_target_final = torch.where(loss_mask, selected_target, torch.full_like(selected_target, -1.0))

    # --- 7) Q-Learning Loss (MSE or Huber) ---
    q_loss = nn.MSELoss()(predicted_q_final, selected_target_final)

    # --------------------------------------------------------------------------
    #  8) POLICY DISTILLATION LOSS: cross-entropy between MCTS distribution and 
    #     the DQN's distribution (softmax of Q-values).
    # --------------------------------------------------------------------------
    distill_loss = torch.tensor(0.0, device=states.device)
    
    if mcts_policy_dist is not None:
        # Convert the policy_net's Q-values to probabilities
        # shape: [B, num_actions]
        # We can apply temperature if we want a "softer" or "sharper" distribution
        dqn_probs = F.softmax(policy_outputs / DISTILL_TEMPERATURE, dim=1)
        
        # For transitions that used MCTS or Hybrid, we want to distill
        # Create a mask: 1 for (mcts or hybrid), 0 for dqn
        if model_used is not None:
            used_mask = torch.tensor([mode_dict.get(mode_used.item(), "dqn") in ("mcts", "hybrid")
                                      for mode_used in model_used],
                                     dtype=torch.bool, device=states.device)
        else:
            # If we don't have model_used, assume all are MCTS transitions
            used_mask = torch.ones(dqn_probs.size(0), dtype=torch.bool, device=states.device)
        
        # We'll compute cross-entropy for the rows in used_mask
        # shape: [B]
        # Cross-entropy for each sample: -sum( p_mcts[i] * log(dqn_probs[i]) )
        ce_all = - (mcts_policy_dist * torch.log(dqn_probs + 1e-8)).sum(dim=1)
        
        # Zero out the entries for non-MCTS transitions
        ce_masked = torch.where(used_mask, ce_all, torch.zeros_like(ce_all))
        
        # Then average over the *entire* batch (or only over used_masked entries)
        # to get a single scalar:
        distill_loss = ce_masked.mean()
    
    # --------------------------------------------------------------------------
    #  9) Final combined loss
    # --------------------------------------------------------------------------
    total_loss = q_loss + ALPHA_DISTILL * distill_loss

    # --- 10) Backprop + optimize ---
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    # Update target net
    soft_update(target_net, policy_net, tau=TAU)

    # Debug info
    if DEBUGMODE:
        print("=== DEBUG: Q-Learning and Distillation ===")
        print(f"Q-Loss: {q_loss.item():.4f} | Distill: {distill_loss.item():.4f} | Total: {total_loss.item():.4f}")
        print("========================================\n")

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
        transitions: list of transitions. Each transition is a tuple:
            (
                state,               # np.array of shape (6,7)
                action,              # int action
                reward,              # float
                next_state,          # np.array of shape (6,7)
                done,                # bool
                best_q_val,          # float or None
                mcts_value,          # float or None
                hybrid_value,        # float or None
                mcts_action,         # int or None
                model_used,          # str or None
                mcts_policy_dist     # list[float] or None
            )
        ep: episode number
        winner: winning player (or -1 for draw)
        total_reward: cumulative reward from player 2 moves
        turn: number of turns (adjusted when player 1 wins)
        mcts_count: count of moves where MCTS was used by player 2
        hybrid_count: count of moves where Hybrid was used by player 2
        dqn_count: count of moves where DQN was used by player 2
    """
    ep, current_epsilon, current_mcts_level, policy_state, logger = args
    
    local_policy_net = DQN().to(DEVICE)
    local_policy_net.load_state_dict(policy_state)
    local_policy_net.eval()
    
    agent = AgentLogic(
        policy_net=local_policy_net,
        device=DEVICE,
        q_threshold=Q_THRESHOLD,
        temperature=TEMPERATURE,
        hybrid_value_threshold=HYBRID_THRESHOLD,
        mcts_simulations=MCTS_SIMULATIONS,
        always_mcts=False,
        always_random=False
    )
    env = Connect4()
    state = env.reset()
    transitions = []
    done = False
    total_reward = 0.0
    winner = None
    turn = 0
    mcts_count = 0
    hybrid_count = 0
    dqn_count = 0

    # For storing info from each turn
    model_used = dqn_action = mcts_action = hybrid_action = rand_action = None
    best_q_val = mcts_value = hybrid_value = None
    mcts_policy_dist = None

    # Variables for self-train evaluation
    evaluate_loaded = False
    evaluator = None

    def switch_model(model_used, mcts_action, rand_action, dqn_action, hybrid_action):
        """Decide which action actually gets played based on model_used."""
        if model_used == "mcts" and mcts_action is not None:
            return mcts_action
        elif model_used == "random" and rand_action is not None:
            return rand_action
        elif model_used == "dqn" and dqn_action is not None:
            return dqn_action
        elif model_used == "hybrid" and hybrid_action is not None:
            return hybrid_action
        return None  # fallback, no valid action

    def get_opponent_action(env, logger, debug=True):
        """
        Determines the action for player 1 based on the opponent type.
        For Self-Train opponents, an evaluator may be loaded on the fly.

        Returns a tuple of:
            (model_used, dqn_action, mcts_action, hybrid_action,
             rand_action, best_q_val, mcts_value, hybrid_value,
             mcts_policy_dist)
        """
        nonlocal evaluate_loaded, evaluator

        # Initialize all return variables to None (or a default value).
        model_used = None
        dqn_action = None
        mcts_action = None
        hybrid_action = None
        rand_action = None
        best_q_val = None
        mcts_value = 0.0
        hybrid_value = None
        mcts_policy_dist = None

        last_mcts_ep = 0
        if episodes_in_current_level > 2000:
            last_mcts_ep = ep
        opponent_type = get_opponent_type(ep, last_mcts_ep)

        if opponent_type == "MCTS":
            sims = current_mcts_level if current_mcts_level is not None else 0
            if sims == 0:
                # Random if no simulations
                rand_action = random.choice(env.get_valid_actions())
                model_used = "random"
            else:
                # MCTS: now we expect select_action to return (a, val, dist)
                mcts_agent = MCTS(logger, num_simulations=sims, debug=True)
                mcts_action, mcts_value, mcts_policy_dist = mcts_agent.select_action(env, env.current_player)
                model_used = "mcts"

            if debug:
                logger.debug(f"Phase: {opponent_type}, MCTS Action SELECT={mcts_action}, MCTS_value={mcts_value}")

        elif opponent_type == "Self-Train":
            # Possibly load evaluator
            if not evaluate_loaded:
                evaluator = AgentLogic(local_policy_net, device=DEVICE, q_threshold=0.8)
                torch.save(local_policy_net.state_dict(), EVAL_MODEL_PATH)
                logger.info(f"Copied local_policy_net into evaluator for evaluation from ep:{ep}.")
                evaluate_loaded = True

            if ep % TARGET_EVALUATE == 0:
                # Use the evaluator's pick_action
                (model_used,
                 dqn_action, mcts_action, hybrid_action,
                 rand_action, best_q_val, mcts_value, hybrid_value, extra) = evaluator.pick_action(
                     env, current_epsilon, logger, debug=True
                 )
            else:
                # Use the main agent's pick_action
                (model_used,
                 dqn_action, mcts_action, hybrid_action,
                 rand_action, best_q_val, mcts_value, hybrid_value, extra) = agent.pick_action(
                     env, current_epsilon, logger, debug=True
                 )

            # If pick_action returns an "extra" dict with MCTS distributions
            if extra is not None:
                mcts_policy_dist = extra.get("mcts_policy_dist", None)

            if debug and all(x is None for x in [dqn_action, mcts_action, hybrid_action, rand_action]):
                logger.debug("Phase: Self-Train, Action SELECT returned no valid action")

        else:
            logger.debug("Unexpected opponent type encountered. Returning default values.")

        return (model_used, dqn_action, mcts_action, hybrid_action,
                rand_action, best_q_val, mcts_value, hybrid_value, mcts_policy_dist)

    # ------------------------------
    # Main loop of the Connect4 game
    # ------------------------------
    while not done:
        turn += 1

        if env.current_player == 1:
            # Player 1's turn (opponent)
            (model_used, dqn_action, mcts_action, hybrid_action,
             rand_action, best_q_val, mcts_value,
             hybrid_value, mcts_policy_dist) = get_opponent_action(env, logger, debug=True)

            # Terminate if no model_used
            if model_used is None:
                return None, None, None, None, None, None, None, None

            action = switch_model(model_used, mcts_action, rand_action, dqn_action, hybrid_action)
            env.make_move(action)

            reward_1, status = agent.compute_reward(env, action, 1)
            next_state = env.get_board().copy()

            done_flag = (status != 0) or env.is_draw()
            transitions.append((
                state, action, reward_1, next_state,
                done_flag, best_q_val, mcts_value, hybrid_value,
                mcts_action, model_used, mcts_policy_dist
            ))
            state = next_state

            if done_flag:
                done = True
                winner = status if status != 0 else -1
                if winner == 1:
                    # Player 1 won => agent (player2) lost => add extra transition
                    reward_2, _ = agent.compute_reward(env, -1, 2)  # negative
                    next_state = env.get_board().copy()

                    transitions.append((
                        state, -1, reward_2, next_state,
                        True, None, None, None, None, model_used, None
                    ))
                    state = next_state
                    total_reward += reward_2
                break

        else:
            # Player2's turn (our agent)
            local_policy_net.eval()

            # pick_action typically returns 8 items
            (model_used, dqn_action, mcts_action, hybrid_action,
             best_q_val, mcts_value, hybrid_value, extra) = agent.pick_action(
                 env, current_epsilon, logger, debug=DEBUGMODE
             )

            # If there's an "extra" dict with MCTS policy dist
            mcts_policy_dist = None
            if extra is not None:
                mcts_policy_dist = extra.get("mcts_policy_dist", None)

            action = switch_model(model_used, mcts_action, None, dqn_action, hybrid_action)

            # Count usage stats
            if model_used == "mcts":
                mcts_count += 1
            elif model_used == "hybrid":
                hybrid_count += 1
            elif model_used == "dqn":
                dqn_count += 1

            env.make_move(action)
            reward_2, status = agent.compute_reward(env, action, 2)
            total_reward += reward_2

            next_state = env.get_board().copy()
            done_flag = (status != 0) or env.is_draw()

            transitions.append((
                state, action, reward_2, next_state,
                done_flag, best_q_val, mcts_value, hybrid_value,
                mcts_action, model_used, mcts_policy_dist
            ))
            state = next_state

            if done_flag:
                done = True
                winner = status if status != 0 else -1
                break

    return transitions, ep, winner, total_reward, turn, mcts_count, hybrid_count, dqn_count


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
        capacity=REPLAYBUFFER_CAPACITY,
        state_shape=(6, 7),
        device=DEVICE,
        version=MODEL_VERSION
    )

    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.RMSprop(policy_net.parameters(),  lr=LR)


    global EPSILON, current_mcts_level, current_level_index, episodes_in_current_level,recent_win_results
    recent_win_results = []
    
    pool = mp.Pool(processes=NUM_WORKERS)

    # Load checkpoint from disk.
    policy_net, target_net, optimizer, replay_buffer, start_ep = load_model_checkpoint(
        MODEL_SAVE_PATH, LR, REPLAYBUFFER_CAPACITY, logger, DEVICE
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
        transitions, ep, winner, total_reward, turn, mcts_count,hybrid_count,dqn_count = async_result.get(timeout=30)
        
        if transitions is None and ep is None and winner is None:
            pool.close()
            pool.join()
            logger.info("Training Completed.")
            print("Training Completed.")
        # Store transitions in replay buffer
        for transition in transitions:
            replay_buffer.push(*transition)

        hybrid_rate=hybrid_count/int(turn/2)
        dqn_rate=dqn_count/int(turn/2)
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
                        f"MCTS Rate:{mcts_rate*100:.2f}%, DQN Rate:{dqn_rate*100:.2f}%, HYBRID Rate:{hybrid_rate*100:.2f}%")
        print(f"Episode {ep}: Winner={winner},Win Rate={current_win_rate*100:.2f}%, Turn={turn}, Reward={total_reward:.2f}, "
                        f"EPSILON={EPSILON:.6f}, MCTS LEVEL={current_mcts_level}, "
                        f"MCTS Rate:{mcts_rate*100:.2f}, DQN Rate:{dqn_rate*100:.2f}%, HYBRID Rate:{hybrid_rate*100:.2f}%")

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
