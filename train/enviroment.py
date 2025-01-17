# environment.py
import numpy as np
import logging

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        self.turn=1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        self.turn=1
        return self.board

    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < 7):
            logging.error(f"Action {action} is out of bounds.")
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action):
        """Place a piece for current_player if valid."""
        if not self.is_valid_action(action):
            logging.error(f"Invalid action {action}.")
            return
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.turn+=1
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Basic 4-in-a-row checks
        board = self.board
        # Horizontal
        for r in range(6):
            for c in range(4):
                if board[r, c] != 0 and np.all(board[r, c:c+4] == board[r, c]):
                    return board[r, c]
        # Vertical
        for r in range(3):
            for c in range(7):
                if board[r, c] != 0 and np.all(board[r:r+4, c] == board[r, c]):
                    return board[r, c]
        # Diag (down-right)
        for r in range(3):
            for c in range(4):
                if board[r, c] != 0 and all(board[r+i, c+i] == board[r, c] for i in range(4)):
                    return board[r, c]
        # Diag (down-left)
        for r in range(3):
            for c in range(3, 7):
                if board[r, c] != 0 and all(board[r+i, c-i] == board[r, c] for i in range(4)):
                    return board[r, c]
        return 0

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [c for c in range(7) if self.is_valid_action(c)]

    def copy(self):
        env_copy = Connect4()
        env_copy.board = self.board.copy()
        env_copy.current_player = self.current_player
        return env_copy

