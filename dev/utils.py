import torch
from collections import deque
# Project-Specific Imports
from connect4 import Connect4  # For the Connect4 game class
from AgentLogic import AgentLogic  # For game logic and AI actions
from DQN import DQN  # For Deep Q-Networks
def load_model_checkpoint(model_path, policy_net, target_net, optimizer, replay_buffer, learning_rate, buffer_size, logger, device):
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
            replay_buffer = deque(maxlen=buffer_size)

        if model_path and torch.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            logger.info(f"Loaded checkpoint from {model_path} at episode {start_episode}")
        else:
            start_episode = 0

    except Exception as e:
        logger.error(f"Error loading checkpoint:{model_path} {e}")
        
        start_episode = 0

    return policy_net, target_net, optimizer, replay_buffer, start_episode

def save_model(model_path, policy_net, optimizer, current_episode, logger):
    try:
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': current_episode
        }, model_path)
        logger.info(f"Model saved at episode {current_episode}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
