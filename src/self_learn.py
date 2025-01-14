import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import logging
import os

from env.connect4 import Connect4
from models.dqn import DQN
from buffers.replay_buffer import DiskReplayBuffer
from agent.agent_mcts import CombinedAgentMCTS
from agent.reward_system import RewardSystem
from utils.logger import setup_logger, safe_make_dir, get_next_index
from utils.train_utils import save_model_checkpoint,periodic_updates,train_agent,load_model_checkpoint

# ------------- Hyperparameters ------------- #
BATCH_SIZE = 32       # Typical batch size > 1 for stable training
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 20000
     
TARGET_UPDATE = 100   # How often to update target net
NUM_EPISODES = 20000

DEV_LEVEL = "TEST"
index = 11
MODEL="Connect4_Agent_Model"
REPLAY_BUFFER_PREFIX = 'replay_buffer'
# Mapping dictionary
paths = {
    "TEST": "..\Test",
    "TRAIN": "..\Train",
    "EVALUATE": "..\Evaluation"
}

def main():
    try:
        global EPSILON, NUM_EPISODES
        path = paths[DEV_LEVEL]
        
        # Define paths
        model_dir = os.path.join(path, "models", str(index))
        log_dir = os.path.join(path, "logs", str(index))
        replay_buffer_dir = os.path.join(path, "replay_buffer")  # Corrected
        
        # Create directories
        safe_make_dir(model_dir)
        safe_make_dir(log_dir)
        safe_make_dir(replay_buffer_dir)  # Ensure this is called
        
        # Define file paths
        MODEL_SAVE_PATH = os.path.join(model_dir, f"{MODEL}.pth")
        log_file_path = os.path.join(log_dir, "log.txt")
        
        # Setup logger
        logger = setup_logger(log_file_path, level=logging.DEBUG)
        logger.info("Logger initialized.")
        
        # Device Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        logger.info(f"Using device: {device}")
        
        # Load or Initialize DQN
        policy_net, target_net, optimizer, replay_buffer, start_ep = load_model_checkpoint(
        MODEL_SAVE_PATH, LEARNING_RATE, REPLAY_BUFFER_SIZE, REPLAY_BUFFER_PREFIX, logger, device
        )
        # Single agent logic with MCTS
        agent_mcts = CombinedAgentMCTS(policy_net)

        # Tracking stats
        win_count_p1 = 0
        win_count_p2 = 0
        draw_count = 0
        logger.info("Initialization Completed.")
        
    except Exception as e:
        logging.error(f"An error occurred in main(): {e}")
        raise

    for episode in range(1, NUM_EPISODES + 1):
        env = Connect4()
        state = env.reset()   # State is a numpy array
        done = False
        current_player = 1
        total_reward1=0
        total_reward2=0
        # Keep track of transitions for the entire episode
        while not done:
            # Pick move for current_player
            action = agent_mcts.pick_action(env, current_player, epsilon=EPSILON, episode=episode)

            if action is None:
                # No valid move => probably board full
                done = True
                reward, win_status = 0.0, -1
                break

            old_state = env.get_board().copy()
            env.make_move(action)  # This flips current_player internally

            # Next state
            new_state = env.get_board().copy()

            # Compute reward from perspective of the player who just moved
            reward, win_status = agent_mcts.compute_reward(env, action, current_player)
            done = (win_status != 0 or env.is_draw())
            if current_player==1:
                total_reward1+=reward
            else:
                total_reward2+=reward
            # Save transition in replay buffer
            replay_buffer.push(old_state, action, reward, new_state, done)

            # Switch turn (already switched inside make_move, so track it here)
            # env.current_player is now the other player
            current_player = env.current_player

        # Identify final result
        final_winner = env.check_winner()
        if final_winner == 1:
            win_count_p1 += 1
            winner_str = "Player 1"
        elif final_winner == 2:
            win_count_p2 += 1
            winner_str = "Player 2"
        elif env.is_draw():
            draw_count += 1
            winner_str = "Draw"
        else:
            # This should not happen, but just in case
            winner_str = "No Winner"

        # Train DQN
        train_agent(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE, GAMMA)

        # Periodically update target net
        EPSILON = periodic_updates(
            episode,
            policy_net, target_net,
            optimizer,
            MODEL_SAVE_PATH,
            EPSILON, EPSILON_MIN, EPSILON_DECAY,
            TARGET_UPDATE, logger
        )
        logger.info(
                    f"Episode {episode}/{NUM_EPISODES}, Epsilon={EPSILON:.4f}, "
                    f"P1 wins={win_count_p1},Total_Reward1={total_reward1}, P2 wins={win_count_p2},Total_Reward2={total_reward2} Draws={draw_count}"
                )
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
