import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
from collections import deque
from connect4 import Connect4  # Assuming Connect4 class is defined in connect4.py
import logging
from DQN import DQN

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 10000
MODEL_SAVE_PATH="./models/Connect4_Agent_Model1.pth"
TRAINER_SAVE_PATH="./trainer/Connect4_Agent_Trainer1.pth"
# Logging configuration
logging.basicConfig(
    filename="log.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Set up devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize two models
policy_net_1 = DQN().to(device)
target_net_1 = DQN().to(device)
policy_net_2 = DQN().to(device)
target_net_2 = DQN().to(device)

# Copy weights from policy nets to target nets
target_net_1.load_state_dict(policy_net_1.state_dict())
target_net_2.load_state_dict(policy_net_2.state_dict())
target_net_1.eval()
target_net_2.eval()

# Optimizers for both agents
optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=LEARNING_RATE)
optimizer_2 = optim.Adam(policy_net_2.parameters(), lr=LEARNING_RATE)

# Replay buffers for both agents
replay_buffer_1 = deque(maxlen=REPLAY_BUFFER_SIZE)
replay_buffer_2 = deque(maxlen=REPLAY_BUFFER_SIZE)

# Training function for a single agent
def train_agent(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return  # Not enough data to train

    # Sample mini-batch from replay buffer
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
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

# Main training loop
def main():
    global EPSILON
    env = Connect4()
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        total_reward_1, total_reward_2 = 0, 0

        while not done:
            # Agent 1's turn
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            q_values_1 = policy_net_1(state_tensor).detach().cpu()

            valid_actions = env.get_valid_actions()
            if valid_actions:
                # Mask invalid actions
                valid_q_values = {action: q_values_1.squeeze()[action] for action in valid_actions}
                action_1 = max(valid_q_values, key=valid_q_values.get)
            else:
                logging.warning("No valid actions available for Agent 1.")
                break

            env.make_move(action_1)
            reward_1 = 1 if env.check_winner() == 1 else -1 if env.check_winner() == 2 else 0
            next_state = env.board.copy()
            done = env.check_winner() != 0 or env.is_draw()
            replay_buffer_1.append((state, action_1, reward_1, next_state, done))
            state = next_state
            total_reward_1 += reward_1

            if done:
                break

            # Agent 2's turn (similar to Agent 1's logic)
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            q_values_2 = policy_net_2(state_tensor).detach().cpu()

            valid_actions = env.get_valid_actions()
            if valid_actions:
                valid_q_values = {action: q_values_2.squeeze()[action] for action in valid_actions}
                action_2 = max(valid_q_values, key=valid_q_values.get)
            else:
                logging.warning("No valid actions available for Agent 2.")
                break

            env.make_move(action_2)
            reward_2 = 1 if env.check_winner() == 2 else -1 if env.check_winner() == 1 else 0
            next_state = env.board.copy()
            done = env.check_winner() != 0 or env.is_draw()
            replay_buffer_2.append((state, action_2, reward_2, next_state, done))
            state = next_state
            total_reward_2 += reward_2

        # Train both agents
        train_agent(policy_net_1, target_net_1, optimizer_1, replay_buffer_1)
        train_agent(policy_net_2, target_net_2, optimizer_2, replay_buffer_2)

        # Update target networks periodically
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # Log progress
        logging.info(
            "Episode %d: Agent 1 Reward: %.2f, Agent 2 Reward: %.2f, EPSILON: %.3f",
            episode, total_reward_1, total_reward_2, EPSILON,
            extra={"episode": episode},
        )

        # Save models periodically
        if episode % TARGET_UPDATE == 0:
            torch.save(policy_net_1.state_dict(), TRAINER_SAVE_PATH)
            torch.save(policy_net_2.state_dict(),MODEL_SAVE_PATH )
            logging.info("Models saved at episode %d", episode)

if __name__ == "__main__":
    main()