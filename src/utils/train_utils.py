import torch
import torch.nn as nn
import os
import torch
import logging
from models.dqn import DQN
from buffers.replay_buffer import DiskReplayBuffer

def train_agent(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # target = r + gamma * max Q(s', a') if not done
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + (1 - dones.float()) * gamma * next_q

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def load_model_checkpoint(model_path, learning_rate, buffer_size, replay_buffer_prefix, logger, device):
    """
    Load the model checkpoint from the given path. Initialize components if not found.

    Args:
        model_path (str): Path to the model checkpoint file.
        learning_rate (float): Learning rate for the optimizer if initializing.
        buffer_size (int): Capacity for the replay buffer if initializing.
        replay_buffer_prefix (str): Prefix path for the replay buffer memmap files.
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
            prefix_path=replay_buffer_prefix,
            device=device
        )
        logger.info("Replay buffer initialized.")
        
        start_episode = 0

        # Check if the checkpoint exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
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
            prefix_path=replay_buffer_prefix,
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
            target_net.load_state_dict(policy_net.state_dict())
            
            logger.info("Target networks updated")

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            save_model(MODEL_SAVE_PATH, policy_net, optimizer, episode, logger)
            logger.info(f"Models saved at episode {episode}")
    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON