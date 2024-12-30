# config.py

import torch

# Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE = 1000
NUM_EPISODES = 100000
TRAINER_SAVE_PATH = 'Connect4_Agent_Trainer3.pth'
MODEL_SAVE_PATH = 'Connect4_Agent_Model3.pth'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
