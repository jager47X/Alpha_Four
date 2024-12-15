import numpy as np
import torch
import tkinter as tk
from tkinter import messagebox
from connect4 import Connect4  # Import the Connect4 class
from AgentLogic import AgentLogic
from TrainAgent import policy_net_2, target_net_2, optimizer_2, replay_buffer_2, train_agent, MODEL_SAVE_PATH

# Global variable for tracking total reward
total_reward = 0

# Connect 4 App
class Connect4App:
    def __init__(self, root, env, model):
        self.root = root
        self.env = env
        self.model = model
        self.model.eval()
        self.canvas = tk.Canvas(root, width=700, height=600, bg="blue")
        self.canvas.pack()
        self.board = env.reset()
        self.cell_size = 100
        self.draw_board()
        self.root.title("Connect 4")
        self.canvas.bind("<Button-1>", self.human_move)

    def draw_board(self):
        """Draw the Connect 4 board."""
        self.canvas.delete("all")
        for row in range(6):
            for col in range(7):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "white"
                if self.board[row, col] == 1:
                    color = "red"
                elif self.board[row, col] == 2:
                    color = "yellow"
                self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill=color)

    def human_move(self, event):
        """Handle the human player's move."""
        col = event.x // self.cell_size
        if col not in self.env.get_valid_actions():
            messagebox.showerror("Invalid Move", "Column is full! Try again.")
            return

        self.env.make_move(col)
        self.board = self.env.board
        self.draw_board()

        if self.env.check_winner() == 1:
            messagebox.showinfo("Game Over", "Congratulations! You win!")
            self.reset_game()
            return
        if self.env.is_draw():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
            return

        self.ai_move()

    def ai_move(self):
        """Handle the AI's move and train the model after each move."""
        global total_reward
        print("\nAI is thinking...")

        # Select the best action using AgentLogic
        action = AgentLogic.combined_action(self.env, policy_net_2)

        # Perform the action in the environment
        self.env.make_move(action)
        next_state = self.env.board.copy()

        # Calculate the reward
        reward = AgentLogic.calculate_reward(self.env, action, current_player=2)
        total_reward += reward
        print(f"AI Reward: {reward}, Total Reward: {total_reward}")

        # Check for terminal state
        done = self.env.check_winner() != 0 or self.env.is_draw()

        # Store the transition in the replay buffer
        replay_buffer_2.append((self.env.board.copy(), action, reward, next_state, done))

        # Train the model if enough samples are in the replay buffer
        train_agent(replay_buffer_2, policy_net_2, target_net_2, optimizer_2)

        # Update the board
        self.board = self.env.board
        self.draw_board()

        # Check for game-end conditions
        if done:
            if self.env.check_winner() == 2:
                messagebox.showinfo("Game Over", "The AI wins! Better luck next time.")
            elif self.env.is_draw():
                messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
        else:
            print(f"AI chose Column {action + 1}")

    def reset_game(self):
        """Reset the game to start over."""
        self.save_model()
        self.board = self.env.reset()
        self.draw_board()

    def save_model(self):
        """Save the trained model."""
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")


def main():
    env = Connect4()

    # Load the model
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH)
        if 'model_state_dict' in checkpoint:
            policy_net_2.load_state_dict(checkpoint['model_state_dict'])
            optimizer_2.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 1)
            print(f"Loaded model from {MODEL_SAVE_PATH}, starting from epoch {start_epoch}.")
        else:
            policy_net_2.load_state_dict(checkpoint)
            print(f"Loaded raw state_dict from {MODEL_SAVE_PATH}.")
    except Exception as e:
        print(f"Failed to load model: {e}. Starting fresh training.")

    # Start the Tkinter application
    root = tk.Tk()
    app = Connect4App(root, env, policy_net_2)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.save_model(), root.destroy()])
    root.mainloop()


if __name__ == "__main__":
    main()
