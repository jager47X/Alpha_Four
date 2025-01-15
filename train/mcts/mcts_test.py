import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import threading
import multiprocessing as mp
from copy import deepcopy

# ---------------------- Connect4 Class ---------------------- #
class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1

    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.columns):
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action):
        """
        Make a move for current_player in the given column.
        This call also flips self.current_player to the other player.
        Returns True if the move was successful, False otherwise.
        """
        if not self.is_valid_action(action):
            return False

        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                self.change_turn()
                return True
        return False  # Column is full

    def change_turn(self):
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Check horizontal, vertical, and diagonal
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row, col:col + 4] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(self.rows - 3):
            for col in range(self.columns):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row:row + 4, col] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if self.board[row, col] != 0 and \
                   all(self.board[row + i, col + i] == self.board[row, col] for i in range(4)):
                    return self.board[row, col]

        for row in range(self.rows - 3):
            for col in range(3, self.columns):
                if self.board[row, col] != 0 and \
                   all(self.board[row + i, col - i] == self.board[row, col] for i in range(4)):
                    return self.board[row, col]
        return 0  # No winner

    def is_draw(self):
        if (self.check_winner() == 0) and np.all(self.board != 0):
            return True
        return False

    def get_valid_actions(self):
        """Return a list of valid columns."""
        return [col for col in range(self.columns) if self.is_valid_action(col)]

    def copy(self):
        new_env = Connect4()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env

# ---------------------- MCTS Classes ---------------------- #
class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state.copy()  # Instance of Connect4
        self.parent = parent
        self.action_taken = action_taken
        self.visits = 0
        self.wins = 0
        self.children = []

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * np.sqrt(np.log(self.parent.visits) / self.visits)

def run_mcts(env, num_simulations=100):
    """
    Simple MCTS implementation with random playouts.
    """
    root = MCTSNode(env)

    for sim in range(1, num_simulations + 1):
        node = root

        # 1) Selection
        while node.children:
            node = max(node.children, key=lambda c: c.ucb())

        # 2) Expansion
        winner_now = node.state.check_winner()
        if winner_now == 0 and not node.state.is_draw():
            valid_actions = node.state.get_valid_actions()
            for act in valid_actions:
                new_state = node.state.copy()
                new_state.make_move(act)
                child = MCTSNode(new_state, parent=node, action_taken=act)
                node.children.append(child)

            # Pick one child at random to rollout
            if node.children:
                node = random.choice(node.children)
                winner_now = node.state.check_winner()

        # 3) Simulation (random playout)
        sim_state = node.state.copy()
        sim_player = sim_state.current_player

        while sim_state.check_winner() == 0 and not sim_state.is_draw():
            vacts = sim_state.get_valid_actions()
            if not vacts:
                break
            choice = random.choice(vacts)
            sim_state.make_move(choice)
            sim_player = sim_state.current_player

        final_winner = sim_state.check_winner()

        # 4) Backpropagation
        current = node
        while current is not None:
            current.visits += 1
            if final_winner == 2:
                current.wins += 1
            elif final_winner == 1:
                current.wins -= 1
            current = current.parent

    # Pick child with max visits
    if not root.children:
        # No expansion => terminal or no valid moves
        valid_actions = env.get_valid_actions()
        if valid_actions:
            chosen_action = random.choice(valid_actions)
            return chosen_action
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action_taken

class PureMCTS:
    def __init__(self, num_simulations=10000, debug=0, num_processes=4):
        self.num_simulations = num_simulations
        self.debug = debug
        self.num_processes = num_processes

    def select_action(self, env):
        if self.debug:
            print(f"Running MCTS with {self.num_simulations} simulations using {self.num_processes} processes.")
        return run_parallel_mcts(env, self.num_simulations, self.num_processes)

# ---------------------- Multiprocessing Setup ---------------------- #
def parallel_mcts_search(args):
    env, num_simulations = args
    return run_mcts(env, num_simulations)

def run_parallel_mcts(env, total_simulations, num_processes):
    simulations_per_process = total_simulations // num_processes
    remaining_simulations = total_simulations % num_processes
    args = [(env.copy(), simulations_per_process) for _ in range(num_processes)]
    if remaining_simulations > 0:
        args.append((env.copy(), remaining_simulations))
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(parallel_mcts_search, args)
    # Aggregate results
    action_counts = {}
    for action in results:
        if action is not None:
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
    # Choose the action with the highest count
    if not action_counts:
        return random.choice(env.get_valid_actions()) if env.get_valid_actions() else None
    best_action = max(action_counts, key=action_counts.get)
    return best_action

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
        action = self.ai.select_action(self.env)
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
    ai = PureMCTS(num_simulations=2000, debug=1, num_processes=num_processes)  # Adjust num_simulations as needed
    root = tk.Tk()
    app = Connect4GUI(root, ai)
    root.mainloop()
