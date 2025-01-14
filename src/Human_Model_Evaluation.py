import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import threading
import multiprocessing as mp
from copy import deepcopy
from agent.mcts import MCTS
from env.connect4 import Connect4
# ---------------------- GUI Class ---------------------- #
class Connect4GUI:
    def __init__(self, root, ai):
        self.root = root
        self.root.title("Connect4")
        self.env = Connect4()
        self.human_player = 1
        self.ai_player = 2
        self.ai = ai  # AI instance
        self.board_buttons = []
        self.move_history = []
        self.create_widgets()

    def create_widgets(self):
        """Create and style the board UI using ttk."""
        style = ttk.Style()
        style.theme_use('clam')  # Use 'clam', 'alt', 'default', or 'classic'
        style.configure('TButton', font=('Helvetica', 12), padding=5)
        style.configure('Human.TButton', background='red')
        style.configure('AI.TButton', background='yellow')

        # Frame for the board
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

        # Frame for controls and indicators
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Player turn indicator
        self.turn_label = ttk.Label(control_frame, text="Your Turn (Red)", font=('Helvetica', 14))
        self.turn_label.pack(side=tk.LEFT, padx=10)

        # AI status indicator
        self.ai_status = ttk.Label(control_frame, text="AI Ready", font=('Helvetica', 14))
        self.ai_status.pack(side=tk.RIGHT, padx=10)

        # Reset button
        reset_button = ttk.Button(control_frame, text="Reset Game", command=self.reset_game)
        reset_button.pack(side=tk.TOP, pady=5)

        # Move history display
        history_label = ttk.Label(control_frame, text="Move History:", font=('Helvetica', 12))
        history_label.pack(side=tk.TOP, pady=(10, 0))
        self.history_text = tk.Text(control_frame, height=5, state='disabled', wrap='word')
        self.history_text.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

    def human_move(self, col):
        """Handle the human move."""
        if self.env.current_player != self.human_player:
            return
        if not self.env.make_move(col):
            messagebox.showwarning("Invalid Move", "Column is full! Choose another one.")
            return

        self.update_board()
        self.move_history.append(f"Human: Column {col+1}")
        self.update_history()
        if self.check_game_over():
            return
        self.update_turn_label()
        # Start AI move in a separate thread
        threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        """Handle the AI move."""
        self.update_ai_status("AI is thinking...")
        action = self.ai.pick_action(self.env)
        if action is not None:
            success = self.env.make_move(action)
            if success:
                self.update_board()
                self.move_history.append(f"AI: Column {action+1}")
                self.update_history()
                if self.check_game_over():
                    return
                self.update_turn_label()
            else:
                messagebox.showwarning("Invalid Move", "AI attempted an invalid move!")
        else:
            messagebox.showinfo("Game Over", "No valid moves left. It's a draw!")
            self.reset_game()
        self.update_ai_status("AI Ready")

    def update_board(self):
        """Update the board UI."""
        for r in range(self.env.rows):
            for c in range(self.env.columns):
                val = self.env.board[r][c]
                self.board_buttons[r][c]['text'] = "X" if val == 1 else "O" if val == 2 else " "
                # Change the button color based on the player
                if val == 1:
                    self.board_buttons[r][c].configure(style='Human.TButton')
                elif val == 2:
                    self.board_buttons[r][c].configure(style='AI.TButton')
                else:
                    self.board_buttons[r][c].configure(style='TButton')

    def check_game_over(self):
        """Check if the game is over."""
        winner = self.env.check_winner()
        if winner:
            player = "Human" if winner == self.human_player else "AI"
            messagebox.showinfo("Game Over", f"{player} wins!")
            self.reset_game()
            return True
        elif self.env.is_draw():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
            return True
        return False

    def reset_game(self):
        """Reset the game."""
        self.env.reset()
        self.update_board()
        self.update_turn_label()
        self.update_ai_status("AI Ready")
        self.move_history = []
        self.update_history()

    def update_turn_label(self):
        """Update the turn indicator label."""
        if self.env.current_player == self.human_player:
            self.turn_label.config(text="Your Turn (Red)")
        else:
            self.turn_label.config(text="AI's Turn (Yellow)")

    def update_ai_status(self, status):
        """Update the AI status indicator."""
        self.ai_status.config(text=status)

    def update_history(self):
        """Update the move history display."""
        self.history_text.config(state='normal')
        self.history_text.delete('1.0', tk.END)
        self.history_text.insert(tk.END, "\n".join(self.move_history))
        self.history_text.config(state='disabled')

# ---------------------- Main Block ---------------------- #
if __name__ == "__main__":
    num_processes = mp.cpu_count()
    ai = agent_
    root = tk.Tk()
    app = Connect4GUI(root, ai)
    root.mainloop()