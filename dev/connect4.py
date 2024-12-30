import numpy as np
import logging
class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 6 rows, 7 columns
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board
    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.board.shape[1]):
            logging.error(f"Action {action} is out of bounds.")
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action,warning):
        if not self.is_valid_action(action) and warning==1:
            logging.error("Invalid action!")
        for row in range(5, -1, -1):  # Start from the bottom row
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
    def changeTurn(self):
        self.current_player = 3 - self.current_player
    def check_winner(self):
        # Check horizontal, vertical, and diagonal for a win
        for row in range(6):
            for col in range(7 - 3):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row, col:col + 4] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(7):
                if self.board[row, col] != 0 and \
                   np.all(self.board[row:row + 4, col] == self.board[row, col]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(7 - 3):
                if self.board[row, col] != 0 and \
                   np.all([self.board[row + i, col + i] == self.board[row, col] for i in range(4)]):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(3, 7):
                if self.board[row, col] != 0 and \
                   np.all([self.board[row + i, col - i] == self.board[row, col] for i in range(4)]):
                    return self.board[row, col]

        return 0  # No winner yet

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [col for col in range(7) if self.is_valid_action(col)]
    
    def copy(self):
        """Return a deep copy of the current environment state."""
        copied_env = Connect4()
        copied_env.board = self.board.copy()
        copied_env.current_player = self.current_player
        return copied_env