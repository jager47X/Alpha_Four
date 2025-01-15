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
    filename='gui_evaluation.log',   # logs go here
    filemode='w',                 # 'w' overwrites each run; use 'a' to append
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
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Flatten from (64,6,7) -> 64 * (6*7) = 2688
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 6, 7]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------ DQNAgent ------------------ #
class DQNAgent:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = DQN(device=self.device).to(self.device)
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

    def pick_action(self, env: Connect4):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)  # shape [7]
        # pick best among valid
        best_action = max(valid_actions, key=lambda a: q_values[a].item())
        return best_action

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
        threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        self.update_ai_status("AI is thinking...")
        action = self.ai.pick_action(self.env)
        if action is not None:
            success = self.env.make_move(action)
            if success:
                logger.info(f"AI -> Column {action+1}")
                self.move_history.append(f"AI: Column {action+1}")

                self.update_board()
                self.update_history()

                if self.check_game_over():
                    self.update_ai_status("AI Ready")
                    return

                self.update_turn_label()
            else:
                messagebox.showwarning("Invalid Move", "AI attempted an invalid move!")
                logger.warning("AI attempted an invalid move!")
        else:
            messagebox.showinfo("Game Over", "No valid moves left. It's a draw!")
            logger.info("Game ended in a draw (no valid moves for AI).")
            self.reset_game()

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
    model_path = "Connect4_Agent_Model.pth"  # Path to your trained DQN
    ai = DQNAgent(model_path, device=device)

    root = tk.Tk()
    app = Connect4GUI(root, ai)
    root.mainloop()

