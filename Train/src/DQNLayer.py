
# ------------- DQN Model ------------- #
class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(6 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 7)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = self.activation(self.bn1(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.bn2(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = self.activation(self.bn3(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = self.dropout2(x)
        x = self.activation(self.bn4(self.fc4(x)) if x.size(0) > 1 else self.fc4(x))
        x = self.activation(self.bn5(self.fc5(x)) if x.size(0) > 1 else self.fc5(x))
        x = self.dropout3(x)
        x = self.activation(self.bn6(self.fc6(x)) if x.size(0) > 1 else self.fc6(x))
        return self.fc7(x)
    # ------------- Training Utilities ------------- #
    def normalize_q_values(q_values):
        if isinstance(q_values, np.ndarray):
            q_values = torch.tensor(q_values, dtype=torch.float32)
        return torch.softmax(q_values, dim=0)

    def train_agent(policy_net, target_net, optimizer, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return  # not enough data

        # Sample from disk-based replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        # Current Q
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target Q
        with torch.no_grad():
            next_q_values = target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones.float()) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def load_model_checkpoint(model_path, policy_net, target_net, optimizer,
                            replay_buffer, learning_rate, buffer_size, logger, device):
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
                # Not used here since we use DiskReplayBuffer
                pass

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                logger.info(f"Checkpoint file path: {model_path} verified.")
                if 'model_state_dict' in checkpoint:
                    policy_net.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_episode = checkpoint.get('episode', 0)
                    logger.info(f"Loaded model from {model_path}, starting from episode {start_episode}.")
                else:
                    policy_net.load_state_dict(checkpoint)
                    start_episode = 0
                    logger.info(f"Loaded raw state_dict from {model_path}. Starting from episode {start_episode}.")
            else:
                logger.error(f"Checkpoint file {model_path} does not exist. Starting fresh.")
                start_episode = 0

        except Exception as e:
            logger.critical(f"Failed to load model from {model_path}: {e}. Starting fresh.")
            policy_net = DQN().to(device)
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
            start_episode = 0

        return policy_net, target_net, optimizer, replay_buffer, start_episode

    def save_model(model_path, policy_net, optimizer, current_episode, logger):
        try:
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': current_episode
            }, model_path)
            logger.info(f"Model checkpoint saved to {model_path} at episode {current_episode}.")
        except Exception as e:
            logger.critical(f"Failed to save model checkpoint to {model_path}: {e}")

    def periodic_updates(
        episode,
        #policy_net_1, target_net_1,
        policy_net_2, target_net_2,
        #optimizer_1,
        optimizer_2,
        #TRAINER_SAVE_PATH,
        MODEL_SAVE_PATH,
        EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
    ):
        try:
            if episode % TARGET_UPDATE == 0:
                target_net_1.load_state_dict(policy_net_1.state_dict())
                target_net_2.load_state_dict(policy_net_2.state_dict())
                logger.info("Target networks updated")

            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            logger.info(f"Epsilon decayed to {EPSILON}")

            if episode % TARGET_UPDATE == 0:
                save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
                save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
                logger.info(f"Models saved at episode {episode}")
        except Exception as e:
            logger.error(f"Error during periodic updates at episode {episode}: {e}")
        return EPSILON