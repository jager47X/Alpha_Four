# ------------- Agent Logic ------------- #
class AgentLogic:
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def get_win_move(self, env, player,debug=1):
        """
        Check for a winning move for the given player before any move is selected.
        Returns the column index of the move or None if no such move exists.
        """
        for col in env.get_valid_actions():
            temp_env = env.copy()
            temp_env.make_move(col, warning=0)  # Simulate the move
            #logging.debug(temp_env.get_board())
            if temp_env.check_winner() == player:
                if debug==1:
                    logging.debug(f"Player {player} can win by placing in column {col}.")
                return col
        if debug==1:
            logging.debug(f"No winning move found for Player {player}.")
        return None

    def get_block_move(self, env, player,debug=1):
        """
        Check for a blocking move to prevent the opponent from winning.
        Returns the column index of the move or None if no such move exists.
        """
        opponent = 3 - player  # Determine the opponent
        valid_actions = env.get_valid_actions()
        if debug==1:
            logging.debug(f"Checking blocking moves for Player {player}. Valid actions: {valid_actions}")

        for col in valid_actions:
            temp_env = env.copy()  # Copy the environment
            temp_env.changeTurn() # change to opponent's move
            temp_env.make_move(col, warning=0)  # Simulate opponent's move
            #logging.debug(temp_env.get_board())

            # Check if the opponent would win in this column
            if temp_env.check_winner() == opponent:
                if debug==1:
                    logging.debug(f"Player {player} can block opponent's win in column {col}.")
                return col  # Block the opponent's winning move
        if debug==1:
            logging.debug(f"No blocking move found for Player {player}.")
        return None

    def monte_carlo_tree_search(self, env, num_simulations=1000):
        # Simplified version from your code to fit here
        class MCTSNode:
            def __init__(self, state, parent=None):
                self.state = state.copy()
                self.parent = parent
                self.children = []
                self.visits = 0
                self.wins = 0

            def ucb_score(self, c=1.414):
                if self.visits == 0:
                    return float('inf')
                return (self.wins / self.visits) + c * np.sqrt(np.log(self.parent.visits) / self.visits)

        root = MCTSNode(env)

        for sim in range(num_simulations):
            node = root
            # Selection
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score())
            # Expansion
            if not node.children and not node.state.check_winner():
                valid_actions = node.state.get_valid_actions()
                for move in valid_actions:
                    temp_env = node.state.copy()
                    temp_env.make_move(move, 0)
                    node.children.append(MCTSNode(temp_env, parent=node))
            # Simulation
            if not node.children:
                continue
            current_state = node.state.copy()
            while not current_state.check_winner() and not current_state.is_draw():
                valid_actions = current_state.get_valid_actions()
                if not valid_actions:
                    break
                move = random.choice(valid_actions)
                current_state.make_move(move, 0)
            winner = current_state.check_winner()
            # Backprop
            current_node = node
            while current_node is not None:
                current_node.visits += 1
                if winner == 2:
                    current_node.wins += 1
                elif winner == 1:
                    current_node.wins -= 1
                current_node = current_node.parent

        if not root.children:
            valid_actions = env.get_valid_actions()
            if valid_actions:
                return random.choice(valid_actions)
            raise RuntimeError("Board is full.")

        best_child = max(root.children, key=lambda n: n.visits)
        best_move = None
        for action, child in zip(env.get_valid_actions(), root.children):
            if child == best_child:
                best_move = action
                break
        if best_move is None:
            valid_actions = env.get_valid_actions()
            best_move = random.choice(valid_actions)
        return best_move
    def logic_based_action(self, env, current_player,debug=1):
        """
        Use logic to decide the move (winning or blocking).
        If no logical move exists, return None.
        """

        # Check for a winning move
        win_move = self.get_win_move(env, current_player,debug)
        if win_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a winning move in column {win_move}.")
            return win_move

        # Check for a blocking move
        block_move = self.get_block_move(env, current_player,debug)
        if block_move is not None:
            if debug==1:
                logging.debug(f"Player {current_player} detected a blocking move in column {block_move}.")
            return block_move

        # No logical move found
        if debug==1:
            logging.debug(f"Player {current_player} found no logical move.")
        return None


    def MCTS_action(self, env, current_episode):
        if current_episode > 10000:
            sims = 10000
        else:
            sims = current_episode
        mcts_action = self.monte_carlo_tree_search(env, sims)
        logging.debug(f"MCTS used level={sims}")
        return mcts_action
      
    def qvalue_action(self, env, current_episode):
        # Q-values
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(self.policy_net.device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()

        # Softmax or any normalization
        q_values = normalize_q_values(q_values)
        valid_actions = env.get_valid_actions()
        valid_qs = {a: q_values[a] for a in valid_actions}
        action = max(valid_qs, key=lambda a: valid_qs[a])
        
        logging.debug(f"SoftMaxed Q-Values: {q_values}")
        return action

    def calculate_reward(self, env, last_action, current_player):
        """
        Calculate the reward for the given player based on the game state and potential outcomes.
        
        Args:
            env: The current game environment.
            action: The action taken by the player.
            current_player: The player (1 or 2) for whom the reward is calculated.
            
        Returns:
            tuple: (reward, win_status), where:
                - reward (float): The calculated reward.
                - win_status (int): 1 if current_player wins, -1 if opponent wins, 0 otherwise.
        """
        opponent = 3 - current_player  # Determine the opponent's player ID

        if env.check_winner() == current_player:
            logging.debug(f"Player {current_player} wins!")
            return 1.0, current_player  # Reward for winning and win status
        elif env.check_winner() == opponent:
            logging.debug(f"Player {current_player} loses!")
            return -1.0, opponent  # Penalty for losing and loss status
        elif env.is_draw():
            logging.debug("Game is a draw!")
            return 0.0, 0  # Neutral reward and no win status
        
        # Additional rewards for strategic moves
        #if self.detect_double_three_in_a_row(env, current_player):
          #  logging.debug(f"Action {last_action} made two '3 in a row' patterns for Player {current_player}.")
          #  return 5.0, 0  # High reward, immediate win
        if self.is_WinBlock_move(env,last_action,current_player):
            logging.debug(f"Action {last_action} is a winning or blocking move for Player {opponent}.")
            return 0.5, 0  # Small reward, no immediate win

        # Small penalty for non-advantageous moves
        return -0.01, 0  # Neutral move, no immediate win


    def detect_double_three_in_a_row(self, env, current_player):
        """
        Detect if the current player has two distinct "3 in a row" patterns on the board.
        Each pattern must have one empty slot that can be filled to create a "4 in a row."
        This indicates an immediate win scenario since blocking one pattern will create another.

        Args:
            env (Connect4): The current game environment.
            current_player (int): The player to check for.

        Returns:
            bool: True if there are two distinct "3 in a row" patterns, False otherwise.
        """
        board = env.get_board()
        potential_winning_columns = set()

        # Check horizontal "3 in a row"
        for row in range(6):  # Iterate over all rows
            for col_start in range(7 - 3):  # Check only ranges where "3 in a row" is possible
                line = board[row, col_start:col_start + 4]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_col = col_start + np.where(line == 0)[0][0]
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Check vertical "3 in a row"
        for col in range(7):  # Iterate over all columns
            for row_start in range(6 - 3):  # Check only ranges where "3 in a row" is possible
                line = board[row_start:row_start + 4, col]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    if env.is_valid_action(col):  # Only valid columns can be considered
                        potential_winning_columns.add(col)

        # Check diagonal (top-left to bottom-right)
        for row_start in range(6 - 3):  # Iterate over possible starting rows
            for col_start in range(7 - 3):  # Iterate over possible starting columns
                line = [board[row_start + i, col_start + i] for i in range(4)]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_idx = np.where(np.array(line) == 0)[0][0]
                    empty_col = col_start + empty_idx
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Check diagonal (bottom-left to top-right)
        for row_start in range(6 - 3):  # Iterate over possible starting rows
            for col_start in range(3, 7):  # Iterate over possible starting columns
                line = [board[row_start + i, col_start - i] for i in range(4)]
                if np.count_nonzero(line == current_player) == 3 and np.count_nonzero(line == 0) == 1:
                    empty_idx = np.where(np.array(line) == 0)[0][0]
                    empty_col = col_start - empty_idx
                    if env.is_valid_action(empty_col):
                        potential_winning_columns.add(empty_col)

        # Return True if at least two distinct potential winning columns are found
        if len(potential_winning_columns) >= 2:
            logging.debug(f"Player {current_player} has two '3 in a row' patterns! Winning columns: {list(potential_winning_columns)}")
            return True

        logging.debug(f"No double '3 in a row' patterns detected for Player {current_player}.")
        return False






    def is_WinBlock_move(self, env, last_move_col, current_player):
        """
        Check if the last move by the opponent resulted in a potential winning or blocking scenario.
        This evaluates the current board after the last move has been made.

        Args:
            env (Connect4): The current game environment.
            last_move_col (int): The column where the last move was made.
            current_player (int): The player to analyze the board for (1 or 2).

        Returns:
            bool: True if the last move created a winning or blocking opportunity, False otherwise.
        """
        board = env.get_board()
        opponent = 3 - current_player

        # Check horizontal, vertical, and diagonal lines around the last move
        last_row = next((row for row in range(6) if board[row, last_move_col] == opponent), None)

        if last_row is None:
            logging.debug(f"No piece found in column {last_move_col} to analyze for win/block.")
            return False

        # Check horizontal win/block
        for col_start in range(max(0, last_move_col - 3), min(7, last_move_col + 1)):
            line = board[last_row, col_start:col_start + 4]
            if np.count_nonzero(line == opponent) == 3 and np.count_nonzero(line == 0) == 1:
                logging.debug(f"Horizontal win/block detected at column {col_start + np.where(line == 0)[0][0]}.")
                return True

        # Check vertical win/block
        if last_row <= 2:  # Only check if enough rows below for a vertical line
            line = board[last_row:last_row + 4, last_move_col]
            if np.count_nonzero(line == opponent) == 3 and np.count_nonzero(line == 0) == 1:
                logging.debug(f"Vertical win/block detected at column {last_move_col}.")
                return True

        # Check diagonal (top-left to bottom-right)
        for offset in range(-3, 1):
            diagonal = [board[last_row + i, last_move_col + i] for i in range(4) 
                        if 0 <= last_row + i < 6 and 0 <= last_move_col + i < 7]
            if len(diagonal) == 4 and np.count_nonzero(diagonal == opponent) == 3 and np.count_nonzero(diagonal == 0) == 1:
                logging.debug(f"Diagonal win/block detected at column {last_move_col + np.where(np.array(diagonal) == 0)[0][0]}.")
                return True

        # Check diagonal (top-right to bottom-left)
        for offset in range(-3, 1):
            diagonal = [board[last_row + i, last_move_col - i] for i in range(4) 
                        if 0 <= last_row + i < 6 and 0 <= last_move_col - i < 7]
            if len(diagonal) == 4 and np.count_nonzero(diagonal == opponent) == 3 and np.count_nonzero(diagonal == 0) == 1:
                logging.debug(f"Diagonal win/block detected at column {last_move_col - np.where(np.array(diagonal) == 0)[0][0]}.")
                return True

        logging.debug("No win/block detected after last move.")
        return False