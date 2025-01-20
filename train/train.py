# train.py
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
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
REPLAY_CAPACITY = 10000
EPSILON = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.001
REPLAY_BUFFER_SIZE = 10000
TARGET_EVALUATE = 100  
TARGET_UPDATE = 100
TOTAL_EPISODES = 100000
RAND_EPISODE_BY = 20000   # use random opp until 20k
MCTS_EPISODE_BY = 50000   # use MCTS opp until 50k
SELF_LEARN_START = 50001
DEBUGMODE = True
EVAL_FREQUENCY = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- File Path ------------- #
LOG_PATH = "./logs"
MODEL_SAVE_PATH = 'Connect4_Agent_Model.pth'     
EVAL_MODEL_PATH = 'Connect4_Agent_EVAL.pth'  

# ------------- Train function ------------- #
def train_step(policy_net, target_net, optimizer, replay_buffer,logger):
    if len(replay_buffer) < BATCH_SIZE:
        print("Replay buffer does not have enough samples.")
        return

    # Sample batch
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = (
        batch["states"],
        batch["actions"],
        batch["rewards"],
        batch["next_states"],
        batch["dones"],
    )

    # Debugging shapes and fixing input
    if len(states.shape) == 5:  # Remove extra dimension if present
        states = states.squeeze(2)
    if len(states.shape) == 3:  # Add channel dimension if missing
        states = states.unsqueeze(1)

    # Ensure next_states has the correct shape
    if len(next_states.shape) == 5:
        next_states = next_states.squeeze(2)
    if len(next_states.shape) == 3:
        next_states = next_states.unsqueeze(1)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        # Compute target Q-values
        next_q = target_net(next_states).max(dim=1)[0]  # Max Q-value for next states
        targets = rewards + (1 - dones.float()) * GAMMA * next_q

    # Compute loss
    loss = nn.MSELoss()(q_values, targets)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional: Log loss for debuggings
    #logger.info(f"Loss: {loss.item()}")




def load_model_checkpoint(model_path, learning_rate, buffer_size, logger, device):
    """
    Load the model checkpoint from the given path. Initialize components if not found.

    Args:
        model_path (str): Path to the model checkpoint file.
        learning_rate (float): Learning rate for the optimizer if initializing.
        buffer_size (int): Capacity for the replay buffer if initializing.
        logger (logging.Logger): Logger instance.
        device (torch.device): Device to load the models onto.

    Returns:
        tuple: (policy_net, target_net, optimizer, replay_buffer, start_episode)
    """
    try:
        # Initialize policy_net
        policy_net = DQN().to(device)
        logger.info("Policy network initialized.")
        
        # Initialize target_net
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        logger.info("Target network initialized.")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        logger.info("Optimizer initialized.")
        
        # Initialize replay_buffer
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.info("Replay buffer initialized.")
        
        start_episode = 0

        # Check if the checkpoint exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"Checkpoint file path: {model_path} verified.")
            
            # Load model state_dict
            if 'policy_net_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                logger.info("Loaded policy network state_dict.")
            else:
                policy_net.load_state_dict(checkpoint)
                logger.info("Loaded raw policy network state_dict.")
            
            # Load optimizer state_dict
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state_dict.")
            
            # Load start_episode
            start_episode = checkpoint.get('episode', 0)
            logger.info(f"Loaded start episode: {start_episode}")
        else:
            logger.warning(f"Checkpoint file {model_path} does not exist. Starting fresh.")

    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh.")
        # Re-initialize everything
        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = DiskReplayBuffer(
            capacity=buffer_size,
            state_shape=(6, 7),
            device=device
        )
        logger.info("Re-initialized policy_net, target_net, optimizer, and replay_buffer.")
        start_episode = 0
    
    return policy_net, target_net, optimizer, replay_buffer, start_episode

def save_model_checkpoint(model_path, policy_net, target_net, optimizer, episode, logger):
    """
    Save the model checkpoint to the given path.

    Args:
        model_path (str): Path to save the model checkpoint file.
        policy_net (DQN): The policy network.
        target_net (DQN): The target network.
        optimizer (torch.optim.Optimizer): The optimizer.
        episode (int): Current episode number.
        logger (logging.Logger): Logger instance.
    """
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

def periodic_updates(
    episode,
    policy_net, target_net,
    optimizer, 
    MODEL_SAVE_PATH,
    EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
):
    try:
        if episode % TARGET_UPDATE == 0:
            save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, episode, logger)
            logger.info(f"Models saved at episode {episode}")
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logger.info("Target networks updated")
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON

# ------------- Main function ------------- #

def main():
    
    global EPSILON, TOTAL_EPISODES,DEBUGMODE

    

    # Configure basic logging
    logging.basicConfig(
        filename='train.log',     # or "./logs/train.log" if you prefer
        filemode='a',             # 'w' overwrites each run; use 'a' to append
        level=logging.DEBUG,      # adjust as needed (e.g., DEBUG, INFO, WARNING)
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create a named logger
    logger = logging.getLogger("Connect4Logger")
    print("Starting Initialization")
    
    # Build replay buffer without prefix_path
    try:
        replay_buffer = DiskReplayBuffer(
            capacity=REPLAY_CAPACITY,
            state_shape=(6,7),
            device=DEVICE
        )
    except Exception as e:
        logger.critical(f"Failed to initialize DiskReplayBuffer: {e}")
        return

    # Build networks
    policy_net = DQN(device=DEVICE).to(DEVICE)
    target_net = DQN(device=DEVICE).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    evaluate_net = DQN(device=DEVICE).to(DEVICE)

    # Load or initialize the networks
    try:
        policy_net, target_net, optimizer, _, start_ep = load_model_checkpoint(
            MODEL_SAVE_PATH, 
            LR, 
            REPLAY_BUFFER_SIZE, 
            logger, 
            DEVICE
        )
    except Exception as e:
        logger.critical(f"Failed to load model checkpoint: {e}")
        return

    # Adjust episodes 
    EPSILON =max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** start_ep))
    logger.info(f"Continue from {start_ep}/ {TOTAL_EPISODES}.")
    print(f"Continue from {start_ep}/ {TOTAL_EPISODES}.")
    logger.info(f"Current Epsilon adjusted to {EPSILON}.")
    print((f"Current Epsilon adjusted to {EPSILON}."))
    # Agent logic
    agent = AgentLogic(policy_net, device=DEVICE, q_threshold=0.5)
    evaluator = AgentLogic(policy_net, device=DEVICE, q_threshold=0.5)

    # Stats
    evaluate_loaded = False
    env = Connect4()
    wins, draws, losses = 0, 0, 0
    print("Starting Initialization is over, now training.")

    endep = None
    for ep in range(start_ep + 1, TOTAL_EPISODES + 1):
        state = env.reset()
        done = False
        endep = ep
        total_reward = 0.0
        winner = None  # Initialize winner for each episode
        turn=1
        while not done:
            
            # Player1's turn
            if env.current_player == 1:
                # Decide opponent type based on episode
                if ep < RAND_EPISODE_BY:
                    opponent_type = "Random"
                elif ep < SELF_LEARN_START:
                    opponent_type = "MCTS"
                else:
                    opponent_type = "Self-Play"

                # If we just entered self-play, freeze a copy
                if opponent_type == "Self-Play" and not evaluate_loaded:
                    evaluate_net.load_state_dict(policy_net.state_dict())
                    evaluate_net.eval()
                    torch.save(evaluate_net.state_dict(), EVAL_MODEL_PATH)
                    evaluator = AgentLogic(policy_net, device=DEVICE, q_threshold=0.5)
                    logger.info(f"Copied policy_net into evaluator_net for evaluation from ep:{ep}.") 
                    evaluate_loaded = True

                def get_opponent_action(env,debug=False):
                    # Random phase => random only
                    if opponent_type == "Random":
                        action=random.choice(env.get_valid_actions())
                        if debug:
                            logging.debug(f"Random Action SELECT={action}")
                        return action

                    # MCTS phase => immediate win/block, then MCTS
                    elif opponent_type == "MCTS":
                        base_sims = 10  # Minimum number of simulations for small episodes
                        scaling_factor =    0.04  # Adjust the growth rate
                        sims = int(base_sims + scaling_factor * ep)  # 0.04 *50000=2000 peak performance
                        sims = min(2000, sims)
                        mcts_action = MCTS(num_simulations=sims, debug=True)
                        action=mcts_action.select_action(env, env.current_player)
                        if debug:
                            logging.debug(f"MCTS Action SELECT={action}")
                        return action
                    
                    elif ep % TARGET_EVALUATE == 0:  # use evaluator to check performance of the current model
                        action=evaluator.pick_action(env, env.current_player, EPSILON, episode=ep, debug=DEBUGMODE)
                        if debug:
                            logging.debug(f"EVALUATE SELECT={action}")
                        return action

                    # Self-play
                    else:
                        action=agent.pick_action(env, env.current_player, EPSILON, episode=ep, debug=DEBUGMODE)
                        if debug:
                            logging.debug(f"SELF Opponent SELECT={action}")
                        return action

                action = get_opponent_action(env,debug=DEBUGMODE)
                env.make_move(action)
                reward, status = agent.compute_reward(env, action, 1)
                # total_reward += reward since we focus on Agent 2 we ignore here

                if (status != 0) or env.is_draw():
                    done = True
                    # Update environment to Q tables
                    next_state = env.get_board().copy()

                    # Push to buffer
                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    # Q-learning step
                    train_step(policy_net, target_net, optimizer, replay_buffer,logger)
                    winner = status if status != 0 else -1  # Adjust based on your environment's convention
                    # Update statistics
                    if winner == 2:
                        wins += 1
                    elif winner == 1:
                        losses += 1
                    elif winner == -1:
                        draws += 1
                    turn=env.turn-1
                    break

            else:  # Player2's turn (always model)
                # Set model to eval mode for single inference
                policy_net.eval()
                action = agent.pick_action(env, env.current_player, EPSILON, episode=ep, debug=DEBUGMODE)
                policy_net.train()

                env.make_move(action)
                reward, status = agent.compute_reward(env, action, 2)

                if ep > SELF_LEARN_START:  # Check if it is self learn phase
                    if ep % TARGET_EVALUATE == 0:  # If so check if that is TARGET_EVALUATE
                        total_reward += reward
                else:
                    total_reward += reward  # if not self learn phase then add reward

                if (status != 0) or env.is_draw():
                    done = True
                    winner = status if status != 0 else -1  # Adjust based on your environment's convention

            
            # Update environment to Q tables
            next_state = env.get_board().copy()

            # Push to buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Q-learning step
            train_step(policy_net, target_net, optimizer, replay_buffer,logger)
            turn=env.turn-1
            # Update statistics
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            elif winner == -1:
                draws += 1
            
        # Periodically update target net and decay epsilon
        EPSILON = periodic_updates(
            ep,
            policy_net, target_net,
            optimizer,
            MODEL_SAVE_PATH,
            EPSILON, EPSILON_MIN, EPSILON_DECAY,
            TARGET_UPDATE, logger
        )

        # Save model checkpoint periodically
        if ep % TARGET_UPDATE == 0:
            try:
                save_model_checkpoint(
                    MODEL_SAVE_PATH,
                    policy_net,
                    target_net,
                    optimizer,
                    ep,
                    logger
                )
            except Exception as e:
                logger.error(f"Failed to save model checkpoint at episode {ep}: {e}")
 
        # Log and print episode summary
        logger.info(
            f"Episode {ep}/{TOTAL_EPISODES}: Winner={winner},Turn={turn}, Reward={total_reward:.2f}, EPSILON={EPSILON:.3f}, (W={wins},D={draws},L={losses})"
        )
        print(
            f"Episode {ep}/{TOTAL_EPISODES}: Winner={winner}, Turn={turn}, Reward={total_reward:.2f}, EPSILON={EPSILON:.3f}, (W={wins}, D={draws}, L={losses})"
        )
    
    # Final save after all episodes
    save_model_checkpoint(MODEL_SAVE_PATH, policy_net, target_net, optimizer, endep, logger)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
