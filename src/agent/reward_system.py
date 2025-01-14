import logging
import math

class RewardSystem:
    """
    Calculates rewards based on game state and actions.
    """

    def calculate_reward(self, env, last_action, current_player):
        """
        Calculate the reward for the given player based on the game state and potential outcomes.
        
        Args:
            env: The current game environment.
            last_action: The action taken by the player.
            current_player: The player (1 or 2) for whom the reward is calculated.  

        Returns:
            tuple: (total_reward, win_status), where:
                - total_reward (float): active_reward - passive_penalty + result_reward
                - result_reward (float): Based on game status
                    * +10.0 if current player wins
                    * -10.0 if opponent wins
                    * +5.0 if draw
                    * 0 otherwise
                - active_reward (float): Based on the single move just made
                - passive_penalty (float): Based on opponent's connected rows
                - win_status (int): 1 if current_player wins, 2 if opponent_player wins, -1 if draw, 0 otherwise.
        """
        board = env.get_board()
        opponent = 3 - current_player

        winner = env.check_winner()
        if winner == current_player:
            # Current player just won
            result_reward = 10.0
            win_status = 1
        elif winner == opponent:
            # Opponent won
            result_reward = -10.0
            win_status = 2
        elif env.is_draw():
            # Draw
            result_reward = 5.0
            win_status = -1
        else:  # Ongoing
            win_status = 0
            result_reward = 0.0

        # Active reward: based on the single move just made
        active_reward = self.get_active_reward(board, last_action, current_player, env)

        # Passive penalty: based on opponent's connected rows
        passive_penalty = self.get_passive_penalty(board, opponent)

        total_reward = result_reward + active_reward - passive_penalty
        return (total_reward, win_status)

    # ----------------------------------------------------
    #  Helper Functions to Detect Patterns, etc.
    # ----------------------------------------------------
    
    def get_active_reward(self, board, last_action, current_player, env):
        """
        Determine the reward from the single move the current_player just made.
        Checks for:
            - Win Move (4 in a row)
            - Block Win Move
            - 3 in a row
            - Block 3 in a row
            - 2 in a row
            - Block 2 in a row
            - Otherwise 0.01
        """
        active_reward=0
        row_played = self.get_row_played(board, last_action)
        if row_played is None:
            logging.debug("No move found in the given column.")
            return 0.0

        # Check immediate 4-in-a-row = "Win Move"
        if self.is_double_threat(board, row_played, current_player):
            active_reward+= 10.0

        # Check if it blocks the opponent's 4-in-a-row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 4):
            active_reward+= 2.5

        # Check 3 in a row
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 3):
            active_reward+= 1.5

        # Check if it blocks opponent's 3 in a row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 3):
            active_reward+= 1.0

        # Check 2 in a row
        if self.causes_n_in_a_row(board, row_played, last_action, current_player, 2):
            active_reward+= 0.5

        # Check if it blocks opponent's 2 in a row
        if self.blocks_opponent_n_in_a_row(board, row_played, last_action, current_player, 2):
            active_reward+= 0.25

        # Otherwise minimal reward
        if active_reward<0.01:
            return 0.01
        return active_reward


    def get_passive_penalty(self, board, opponent):
        """
        Calculate a "passive" penalty based on the opponent's connected rows:
            - Count total 2-in-a-rows => each adds 0.1
            - Count total 3-in-a-rows => each adds 1.0
        """
        two_in_a_rows = self.count_n_in_a_row(board, opponent, 2)
        three_in_a_rows = self.count_n_in_a_row(board, opponent, 3)

        # 2 in a row => 0.1 per
        # 3 in a row => 1.0 per
        passive_penalty = two_in_a_rows * 0.1 + three_in_a_rows * 1.0
        return passive_penalty

    def get_row_played(self, board, col):
        """
        Returns the row index where the last piece in 'col' is placed.
        If no piece is found, return None.
        """
        rows = board.shape[0]
        for r in range(rows):
            if board[r, col] != 0:  # Found a piece
                return r
        return None
    def is_double_threat(self,board, col_to_place, current_player):
        """
        Check if placing a piece in 'col_to_place' for 'current_player'
        creates at least two distinct next-move wins (i.e., a double threat).
        Returns True if it's a double threat, otherwise False.
        """

        # 1) Copy the board so we don’t alter the original
        temp_board = board.copy()
        # 2) Place the piece for current_player in col_to_place
        if not self.place_piece(temp_board, col_to_place, current_player):
            # If we can't place (column full or invalid), it's not a threat
            return False

        # 3) Now, for the *next* turn (imagine you move again immediately),
        #    check how many columns produce a win if the current_player places there.
        winning_moves = 0
        for c in self.find_valid_columns(temp_board):
            # Copy again so we don’t ruin temp_board
            next_board = temp_board.copy()
            if self.place_piece(next_board, c, current_player):
                if self.check_if_winning_move(next_board, current_player):
                    winning_moves += 1
                # revert is implicitly done by discarding next_board

            # If we ever find 2 or more winning moves => double threat
            if winning_moves >= 2:
                return True

        return False


    def place_piece(self,board, col, player):
        """
        Drop 'player'’s piece in 'col' on 'board'.
        Return True if successful; False if column is invalid/full.
        """
        if col < 0 or col >= board.shape[1] or board[0, col] != 0:
            return False  # invalid
        rows = board.shape[0]
        for row in range(rows-1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return True
        return False  # column was full, should not happen if checked properly


    def find_valid_columns(self,board):
        """Return columns (0..6) that are not full."""
        valid_cols = []
        for col in range(board.shape[1]):
            if board[0, col] == 0:
                valid_cols.append(col)
        return valid_cols


    def check_if_winning_move(self,board, player):
        """
        Return True if 'player' has a 4-in-a-row on 'board'.
        Uses your existing 4-in-a-row logic or a smaller version here.
        """
        # For brevity, just re-use your Connect4 logic or a quick local check.
        return self.four_in_a_row_exists(board, player)


    def four_in_a_row_exists(self,board, player):
        # Implement horizontal, vertical, diagonal checks for 'player'.
        # Return True if there's a connect-4, else False.
        # For brevity, assume a simplified snippet or your existing function:
        rows, cols = board.shape
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r, c+1] == board[r, c+2] == board[r, c+3]):
                    return True
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (board[r, c] == player == board[r+1, c] == board[r+2, c] == board[r+3, c]):
                    return True
        # Diagonal \
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (board[r, c] == player == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]):
                    return True
        # Diagonal /
        for r in range(rows - 3):
            for c in range(3, cols):
                if (board[r, c] == player == board[r+1, c-1] == board[r+2, c-2] == board[r+3, c-3]):
                    return True
        return False

    def causes_n_in_a_row(self, board, row, col, player, n):
        """
        Check if placing at (row, col) caused 'player' to have n in a row.
        """
        return (
            self.check_line(board, row, col, player, n, "horizontal") or
            self.check_line(board, row, col, player, n, "vertical")   or
            self.check_line(board, row, col, player, n, "diag1")      or
            self.check_line(board, row, col, player, n, "diag2")
        )

    def blocks_opponent_n_in_a_row(self, board, row, col, current_player, n):
        """
        Temporarily remove current_player's piece and check if 
        that spot would have caused the opponent to get 'n' in a row.
        """
        opponent = 3 - current_player
        original_value = board[row, col]
        board[row, col] = 0
        caused = self.causes_n_in_a_row(board, row, col, opponent, n)
        board[row, col] = original_value  # Restore
        return caused

    def check_line(self, board, row, col, player, n, direction):
        """
        Check if there's a continuous line of length >= n for 'player' that includes (row, col),
        in the given direction.
        """
        rows, cols = board.shape

        def count_consecutive(r_step, c_step):
            count = 1  # Include current cell
            # Forward
            rr, cc = row + r_step, col + c_step
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                count += 1
                rr += r_step
                cc += c_step
            # Backward
            rr, cc = row - r_step, col - c_step
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                count += 1
                rr -= r_step
                cc -= c_step
            return count

        if direction == "horizontal":
            return count_consecutive(0, 1) >= n
        elif direction == "vertical":
            return count_consecutive(1, 0) >= n
        elif direction == "diag1":  # Top-left -> Bottom-right
            return count_consecutive(1, 1) >= n
        elif direction == "diag2":  # Top-right -> Bottom-left
            return count_consecutive(1, -1) >= n

    def count_n_in_a_row(self, board, player, n):
        """
        Count how many distinct sequences of length >= n for 'player' exist on the board.
        """
        rows, cols = board.shape
        visited_segments = set()
        total_count = 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def in_bounds(r, c):
            return 0 <= r < rows and 0 <= c < cols

        # Check each cell for possible lines
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == player:
                    for dr, dc in directions:
                        cells_in_line = []
                        rr, cc = r, c
                        # Collect continuous sequence in the forward direction
                        while in_bounds(rr, cc) and board[rr, cc] == player:
                            cells_in_line.append((rr, cc))
                            rr += dr
                            cc += dc
                        # If the length is at least n, we might have multiple overlapping segments
                        if len(cells_in_line) >= n:
                            # Slide a window of length n across the line
                            for start_idx in range(len(cells_in_line) - n + 1):
                                segment = tuple(cells_in_line[start_idx:start_idx + n])
                                if segment not in visited_segments:
                                    visited_segments.add(segment)
                                    total_count += 1
        return total_count
