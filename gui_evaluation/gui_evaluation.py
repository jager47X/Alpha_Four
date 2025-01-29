import math
import random
import logging
import os
import threading
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tkinter as tk
from tkinter import messagebox, ttk

# ----------------- Logging Setup ----------------- #
logging.basicConfig(
    filename='gui_evaluation.log',  # logs go here
    filemode='w',                  # 'w' overwrites each run; use 'a' to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Connect4Logger")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ Connect4 Environment ------------------ #
class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.columns):
            logger.error(f"Action {action} is out of bounds.")
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action, warning=1):
        """Returns True if valid, False if column is full or out of range."""
        if not self.is_valid_action(action):
            if warning == 1:
                logger.error("Invalid action!")
            return False
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                self.current_player = 3 - self.current_player
                return True
        return False

    def check_winner(self):
        # horizontal
        for r in range(self.rows):
            for c in range(self.columns - 3):
                val = self.board[r, c]
                if val != 0 and np.all(self.board[r, c:c+4] == val):
                    return val
        # vertical
        for r in range(self.rows - 3):
            for c in range(self.columns):
                val = self.board[r, c]
                if val != 0 and np.all(self.board[r:r+4, c] == val):
                    return val
        # diagonal \
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                val = self.board[r, c]
                if val != 0 and all(self.board[r+i, c+i] == val for i in range(4)):
                    return val
        # diagonal /
        for r in range(self.rows - 3):
            for c in range(3, self.columns):
                val = self.board[r, c]
                if val != 0 and all(self.board[r+i, c-i] == val for i in range(4)):
                    return val
        return 0

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [col for col in range(self.columns) if self.is_valid_action(col)]

    def copy(self):
        new_env = Connect4()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env

# ------------------ Minimal DQN ------------------ #
class DQN(nn.Module):
    def __init__(self, input_shape=(6, 7), num_actions=7, dropout_prob=0.3, device=None):
        """
        A deeper and wider DQN for Connect 4.

        Args:
            input_shape (tuple): Shape of the input (rows, columns).
            num_actions (int): Number of possible actions (e.g., columns in Connect 4).
            dropout_prob (float): Dropout probability for regularization.
            device (torch.device): Device to use for computations (CPU or GPU).
        """
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_actions = num_actions

        # 1) Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 6, 7)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 6, 7)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 6, 7)
        self.bn3 = nn.BatchNorm2d(128)

        # 2) Flattening Layer and Fully Connected Layers
        # After conv3, output shape is (batch_size, 128, 6, 7) => Flatten to (batch_size, 128 * 6 * 7)
        flattened_size = 128 * input_shape[0] * input_shape[1]  # 128 * 6 * 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_actions)  # Output: Q-values for each action

        # Dropouts for regularization
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, rows, columns].

        Returns:
            torch.Tensor: Q-values for each action of shape [batch_size, num_actions].
        """
        # Ensure input has the correct shape: [batch_size, 1, rows, columns]
        if len(x.shape) == 5:  # Case: [batch_size, 1, 1, rows, columns]
            x = x.squeeze(2)
        if len(x.shape) == 3:  # Case: [batch_size, rows, columns]
            x = x.unsqueeze(1)  # Add channel dimension to make [batch_size, 1, rows, columns]

        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 32, rows, columns)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, rows, columns)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 128, rows, columns)

        # Flatten and fully connected layers
        x = self.flatten(x)                  # (batch_size, 128 * rows * columns)
        x = F.relu(self.bn4(self.fc1(x)))    # Fully connected layer 1
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))    # Fully connected layer 2
        x = self.dropout2(x)
        x = F.relu(self.bn6(self.fc3(x)))    # Fully connected layer 3
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)                      # Output: (batch_size, num_actions)
        return x

# ------------------ DQNAgent with Real-Time Training ------------------ #
class DQNAgent:
    def __init__(self, model_path, device="cpu", epsilon=0.1, q_threshold=0.5, lr=0.01):
        """
        Args:
            model_path (str): Path to the saved model.
            device (str or torch.device): 'cpu' or 'cuda'.
            epsilon (float): Probability of choosing a random action.
            q_threshold (float): If best Q-value (softmax-based) is below this, pick random.
            lr (float): Learning rate for on-the-fly training.
        """
        self.device = device
        self.model = DQN(device=self.device).to(self.device)
        # Try to load existing weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded DQN model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found; using uninitialized weights.")
        self.model.eval()

        self.epsilon = epsilon
        self.q_threshold = q_threshold

        # Optimizer & loss for real-time learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = 0.99  # discount factor

    def pick_action(self, env: Connect4):
        """
        Epsilon-greedy + threshold-based:
          1) With probability epsilon, pick random valid action.
          2) Otherwise, use Q-network. Convert to softmax, pick best valid.
          3) If that best Q-value is < q_threshold, pick random.
        """
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        # 1) Epsilon random
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 2) Evaluate Q-network
        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # Convert to softmax "confidence"
        soft_q = self._softmax_q(q_values).numpy()
        print(soft_q)
        # Among valid actions, pick the highest
        best_action = max(valid_actions, key=lambda a: soft_q[a])
        best_val = soft_q[best_action]

        # 3) If below threshold => random
        if best_val < self.q_threshold:
            return random.choice(valid_actions)

        return best_action

    def train_on_transition(self, state, action, reward, next_state, done):
        # Force BN layers to use eval mode
        was_training = self.model.training
        self.model.train()  # put model in train mode for dropout, etc.
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

        # Convert to tensors
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_vals = self.model(state_t)
        q_val = q_vals[0, action]

        with torch.no_grad():
            q_next = self.model(next_state_t)
            max_q_next = torch.max(q_next)

        if done:
            target = torch.tensor([reward], dtype=torch.float32, device=self.device)
        else:
            target = torch.tensor([reward], dtype=torch.float32, device=self.device) + self.gamma * max_q_next

        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Restore model's original train/eval state if needed
        if not was_training:
            self.model.eval()

    def _softmax_q(self, q_values):
        """Utility to convert array of Q-values into a softmax distribution."""
        q_tensor = torch.tensor(q_values, dtype=torch.float32)
        return F.softmax(q_tensor, dim=0)


# ------------------ GUI (Human vs DQN) ------------------ #
class Connect4GUI:
    def __init__(self, root, ai):
        self.root = root
        self.root.title("Connect4")
        self.env = Connect4()
        self.human_player = 1
        self.ai_player = 2
        self.ai = ai
        self.board_buttons = []
        self.move_history = []

        self.create_widgets()
        logger.info("New Connect4 game started: Human (Red) vs AI (Yellow)")

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=5)
        style.configure('Human.TButton', background='red')
        style.configure('AI.TButton', background='yellow')

        board_frame = ttk.Frame(self.root, padding="10")
        board_frame.pack(side=tk.TOP, expand=True)

        for row in range(self.env.rows):
            row_buttons = []
            for col in range(self.env.columns):
                button = ttk.Button(
                    board_frame,
                    text=" ",
                    width=4,
                    command=lambda c=col: self.human_move(c)
                )
                button.grid(row=row, column=col, padx=2, pady=2)
                row_buttons.append(button)
            self.board_buttons.append(row_buttons)

        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.turn_label = ttk.Label(control_frame, text="Your Turn (Red)", font=('Helvetica', 14))
        self.turn_label.pack(side=tk.LEFT, padx=10)

        self.ai_status = ttk.Label(control_frame, text="AI Ready", font=('Helvetica', 14))
        self.ai_status.pack(side=tk.RIGHT, padx=10)

        reset_button = ttk.Button(control_frame, text="Reset Game", command=self.reset_game)
        reset_button.pack(side=tk.TOP, pady=5)

        history_label = ttk.Label(control_frame, text="Move History:", font=('Helvetica', 12))
        history_label.pack(side=tk.TOP, pady=(10, 0))

        self.history_text = tk.Text(control_frame, height=5, state='disabled', wrap='word')
        self.history_text.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

    def human_move(self, col):
        if self.env.current_player != self.human_player:
            return

        success = self.env.make_move(col)
        if not success:
            messagebox.showwarning("Invalid Move", "Column is full or invalid! Choose another one.")
            return

        logger.info(f"Human -> Column {col+1}")
        self.move_history.append(f"Human: Column {col+1}")
        self.update_board()
        self.update_history()

        if self.check_game_over():
            return

        self.update_turn_label()
        # AI move in a background thread
        threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        self.update_ai_status("AI is thinking...")

        # ------ 1) Capture the old state before action ------
        old_state = np.copy(self.env.board)

        # ------ 2) AI picks an action ------
        action = self.ai.pick_action(self.env)
        if action is None:
            messagebox.showinfo("Game Over", "No valid moves left. It's a draw!")
            logger.info("Game ended in a draw (no valid moves for AI).")
            self.reset_game()
            return

        # ------ 3) Environment transitions ------
        success = self.env.make_move(action)
        if not success:
            messagebox.showwarning("Invalid Move", "AI attempted an invalid move!")
            logger.warning("AI attempted an invalid move!")
            self.update_ai_status("AI Ready")
            return

        logger.info(f"AI -> Column {action+1}")
        self.move_history.append(f"AI: Column {action+1}")
        self.update_board()
        self.update_history()

        # Check if game ended
        winner = self.env.check_winner()
        done = (winner != 0 or self.env.is_draw())

        # Simple reward scheme:
        # +1 if AI just won, -1 if AI lost, 0 if draw or not finished
        if winner == self.ai_player:
            reward = 1.0
        elif winner == self.human_player:
            reward = -1.0
        elif self.env.is_draw():
            reward = 0.0
        else:
            reward = 0.0

        new_state = np.copy(self.env.board)

        # ------ 4) Train the agent with this single transition ------
        self.ai.model.train()  # Ensure model is in training mode
        self.ai.train_on_transition(old_state, action, reward, new_state, done)
        self.ai.model.eval()   # Switch back to eval mode

        # If the game is over, show message and reset
        if done:
            if winner == self.ai_player:
                messagebox.showinfo("Game Over", "AI (Yellow) wins!")
                logger.info("GAME OVER => Winner: AI (Yellow)")
            elif winner == self.human_player:
                messagebox.showinfo("Game Over", "Human (Red) wins!")
                logger.info("GAME OVER => Winner: Human (Red)")
            else:
                messagebox.showinfo("Game Over", "It's a draw!")
                logger.info("GAME OVER => Draw")

            self.reset_game()
            self.update_ai_status("AI Ready")
            return

        # Otherwise, continue the game
        self.update_turn_label()
        self.update_ai_status("AI Ready")

    def update_board(self):
        for r in range(self.env.rows):
            for c in range(self.env.columns):
                val = self.env.board[r][c]
                if val == 1:
                    self.board_buttons[r][c]['text'] = "X"
                    self.board_buttons[r][c].configure(style='Human.TButton')
                elif val == 2:
                    self.board_buttons[r][c]['text'] = "O"
                    self.board_buttons[r][c].configure(style='AI.TButton')
                else:
                    self.board_buttons[r][c]['text'] = " "
                    self.board_buttons[r][c].configure(style='TButton')

    def check_game_over(self):
        winner = self.env.check_winner()
        if winner != 0:
            if winner == self.human_player:
                winner_str = "Human (Red)"
            else:
                winner_str = "AI (Yellow)"
            messagebox.showinfo("Game Over", f"{winner_str} wins!")
            logger.info(f"GAME OVER => Winner: {winner_str}")
            self.reset_game()
            return True
        elif self.env.is_draw():
            messagebox.showinfo("Game Over", "It's a draw!")
            logger.info("GAME OVER => Draw")
            self.reset_game()
            return True
        return False

    def reset_game(self):
        logger.info("Resetting game...")
        self.env.reset()
        self.update_board()
        self.update_turn_label()
        self.update_ai_status("AI Ready")
        self.move_history.clear()
        self.update_history()

    def update_turn_label(self):
        if self.env.current_player == self.human_player:
            self.turn_label.config(text="Your Turn (Red)")
        else:
            self.turn_label.config(text="AI's Turn (Yellow)")

    def update_ai_status(self, status):
        self.ai_status.config(text=status)

    def update_history(self):
        self.history_text.config(state='normal')
        self.history_text.delete('1.0', tk.END)
        self.history_text.insert(tk.END, "\n".join(self.move_history))
        self.history_text.config(state='disabled')

# --------------- Main --------------- #
if __name__ == "__main__":
    model_path = "Connect4_Agent_Model1.pth"  # Path to your trained DQN (if any)

    # A high learning rate for "max weight of learning"; be cautious—this may be unstable.
    ai = DQNAgent(model_path, device=device, epsilon=0.1, q_threshold=0.5, lr=0.01)

    root = tk.Tk()
    app = Connect4GUI(root, ai)
    root.mainloop()
