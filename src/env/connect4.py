import numpy as np

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board

    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.board.shape[1]):
            return False
        if self.board[0, action] != 0:
            return False
        return True

    def make_move(self, action):
        """
        Make a move for current_player in the given column.
        This call also flips self.current_player to the other player.
        """
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.change_turn()

    def change_turn(self):
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Check horizontal, vertical, and diagonal
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
                   all(self.board[row + i, col + i] == self.board[row, col] for i in range(4)):
                    return self.board[row, col]

        for row in range(6 - 3):
            for col in range(3, 7):
                if self.board[row, col] != 0 and \
                   all(self.board[row + i, col - i] == self.board[row, col] for i in range(4)):
                    return self.board[row, col]
        return 0  # No winner

    def is_draw(self):
        return (self.check_winner() == 0) and np.all(self.board != 0)

    def get_valid_actions(self):
        """Return a list of valid columns."""
        return [col for col in range(7) if self.is_valid_action(col)]

    def copy(self):
        new_env = Connect4()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env
