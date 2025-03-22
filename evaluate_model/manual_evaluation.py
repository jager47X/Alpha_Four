import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import torch
import warnings
from numba.core.errors import NumbaPerformanceWarning

from dependencies.environment import Connect4
from dependencies.models import DQN
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import setup_logger, safe_make_dir, get_next_index
from dependencies.mcts import MCTS

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Logging Setup ----------------- #

# Define the log directory and log file path
log_dir = os.path.join("data", "logs", "evaluation_logs")
log_file = os.path.join(log_dir, "manual_evaluation.log")

# Ensure the log directory exists before logging
os.makedirs(log_dir, exist_ok=True)

# Initialize logging
logging.basicConfig(
    filename=log_file,  # Logs go here
    filemode="w",       # 'w' overwrites each run; use 'a' to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)  # Initialize logger before use
logger.info("Logging setup complete!")

# ----------------- Initialization ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ GUI (Human vs AI) ------------------ #
class Connect4GUI:
    def __init__(self, root, ai):
        self.root = root
        self.root.title("Connect4")
        # Use the FixedConnect4 subclass instead of the original Connect4
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

        # Determine board dimensions from self.env.board
        num_rows = len(self.env.board)
        num_cols = len(self.env.board[0]) if num_rows > 0 else 0

        for row in range(num_rows):
            row_buttons = []
            for col in range(num_cols):
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

        # Check if game over after human move.
        if self.check_game_over():
            winner = self.env.check_winner()
            if winner != 0:
                winner_str = "Human (Red)" if winner == self.human_player else "AI (Yellow)"
            else:
                winner_str = "Draw"
            messagebox.showinfo("Game Over", f"{winner_str} wins!")
            logger.info(f"GAME OVER => Winner: {winner_str}")
            self.reset_game()
            return

        self.update_turn_label()
        # AI move in a background thread
        threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        self.update_ai_status("AI is thinking...")

        # Expect pick_action to return (action, q_value, extra)
        action, q_value = self.ai.pick_action(
            self.env, epsilon=0.1, logger=logger, debug=True, mcts_fallback=True,evaluation=True
        )

        if action is None:
            self.root.after(0, lambda: messagebox.showinfo("Game Over", "No valid moves left. It's a draw!"))
            logger.info("Game ended in a draw (no valid moves for AI).")
            self.root.after(0, self.reset_game)
            return

        success = self.env.make_move(action)
        if not success:
            self.root.after(0, lambda: messagebox.showwarning("Invalid Move", "AI attempted an invalid move!"))
            logger.warning("AI attempted an invalid move!")
            self.update_ai_status("AI Ready")
            return

        if q_value is not None:
            logger.info(f"AI -> Column {action+1} (Q-Value: {q_value:.3f})")
        self.move_history.append(f"AI: Column {action+1}")
        self.update_board()
        self.update_history()

        # Immediately check if AI win occurred.
        winner = self.env.check_winner()
        done = (winner != 0 or self.env.is_draw())


        self.ai.policy_net.train()  # Ensure model is in training mode
        self.ai.policy_net.eval()  # Switch back to eval mode

        # If game is over, immediately schedule the game-over handler on the main thread
        if done:
            self.root.after(0, self.handle_game_over, winner)
            return

        self.update_turn_label()
        self.update_ai_status("AI Ready")

    def handle_game_over(self, winner):
        if winner != 0:
            winner_str = "AI (Yellow)" if winner == self.ai_player else "Human (Red)"
        else:
            winner_str = "Draw"
        messagebox.showinfo("Game Over", f"{winner_str} wins!")
        logger.info(f"GAME OVER => Winner: {winner_str}")
        self.reset_game()

    def update_board(self):
        num_rows = len(self.env.board)
        num_cols = len(self.env.board[0]) if num_rows > 0 else 0
        for r in range(num_rows):
            for c in range(num_cols):
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
        return (winner != 0 or self.env.is_draw())

    def reset_game(self):
        logger.info("Resetting game...")
        self.env.reset()
        self.update_board()
        self.update_turn_label()
        self.update_ai_status("AI Ready")
        self.move_history.clear()
        self.update_history()

    def update_turn_label(self):
        self.turn_label.config(
            text="Your Turn (Red)" if self.env.current_player == self.human_player else "AI's Turn (Yellow)"
        )

    def update_ai_status(self, status):
        self.ai_status.config(text=status)

    def update_history(self):
        self.history_text.config(state='normal')
        self.history_text.delete('1.0', tk.END)
        self.history_text.insert(tk.END, "\n".join(self.move_history))
        self.history_text.config(state='disabled')

# --------------- Main --------------- #
if __name__ == "__main__":
    model_version = input("model_version>> ") or 2
    print("main_model_version: ", model_version)
    MODEL_PATH = os.path.join("data", "models", str(model_version), "Connect4_Agent_Model.pth")
    print(f"\nLoading Main DQN Agent from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    policy_net = DQN().to(device)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent_main = AgentLogic(policy_net, device)
    ai = AgentLogic(policy_net , device)
    
    root = tk.Tk()
    app = Connect4GUI(root, ai)
    root.mainloop()
