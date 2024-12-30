import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
from connect4 import Connect4  # Import the Connect4 class
import logging
import random
from copy import deepcopy
# Project-Specific Imports
from logger_utils import setup_logger
from connect4 import Connect4  # For the Connect4 game class
from AgentLogic import AgentLogic  # For game logic and AI actions
from DQN import DQN  # For Deep Q-Networks
import logging
from config import (
    BATCH_SIZE,
    GAMMA,
    device,
)
def train_agent(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        logging.info("Not enough data to train")
        return  # Not enough data to train

    # Sample mini-batch from replay buffer
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    # Convert the list of numpy arrays to a single numpy ndarray before creating the tensor
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)

    dones = torch.tensor(dones, dtype=torch.bool).to(device)


    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    next_q_values = target_net(next_states).max(1)[0]
    targets = rewards + (1 - dones.float()) * GAMMA * next_q_values

    # Compute loss and optimize
    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()