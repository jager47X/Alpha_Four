import numpy as np
import torch
import torch.nn as nn
import logging
from copy import deepcopy
import random
from copy import deepcopy
from collections import deque
import torch.optim as optim
from connect4 import Connect4  # Import the Connect4 class
import logging
import random
from copy import deepcopy
import torch


class AgentLogic:
    def __init__(self, policy_net, device):
        """
        Initialize the AgentLogic class with a policy network and device.
        
        :param policy_net: The neural network model for predicting Q-values.
        :param device: The computation device (e.g., 'cuda' or 'cpu').
        """
        self.policy_net = policy_net
        self.device = device

    def get_win_move(self, env, player):
        """
        Check for a winning move for the given player.
        Returns the column index of the move or None if no such move exists.
        """
        for col in env.get_valid_actions():
            temp_env = deepcopy(env)
            temp_env.current_player = player
            temp_env.make_move(col)
            if temp_env.check_winner() == player:
                return col
        return None

    def get_block_move(self, env, player):
        """
        Check for a blocking move to prevent the opponent from winning.
        Returns the column index of the move or None if no such move exists.
        """
        opponent = 3 - player
        for col in env.get_valid_actions():
            temp_env = deepcopy(env)
            temp_env.current_player = opponent
            temp_env.make_move(col)
            if temp_env.check_winner() == opponent:
                return col
        return None

    def logic_based_action(self, env):
        """
        Use logic to decide the move (winning or blocking).
        If no logical move exists, return None.
        """
        win_move = self.get_win_move(env, player=2)
        if win_move is not None:
            return win_move

        block_move = self.get_block_move(env, player=1)
        if block_move is not None:
            return block_move

        return None

    def monte_carlo_tree_search(self, env, num_simulations=100):
        """
        Monte Carlo Tree Search for decision-making in Connect4.
        Returns the column index of the best move.
        """
        class MCTSNode:
            def __init__(self, state, parent=None):
                self.state = deepcopy(state)
                self.parent = parent
                self.children = []
                self.visits = 0
                self.wins = 0

            def ucb_score(self, exploration_constant=1.414):
                if self.visits == 0:
                    return float('inf')
                win_rate = self.wins / self.visits
                exploration_term = exploration_constant * \
                    (np.sqrt(np.log(self.parent.visits) / self.visits))
                return win_rate + exploration_term

        root = MCTSNode(env)

        for _ in range(num_simulations):
            node = root
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score())

            if not node.children and not node.state.check_winner():
                for move in node.state.get_valid_actions():
                    temp_env = deepcopy(node.state)
                    temp_env.make_move(move)
                    child_node = MCTSNode(temp_env, parent=node)
                    node.children.append(child_node)

            current_node = node
            current_state = deepcopy(current_node.state)
            while not current_state.check_winner() and not current_state.is_draw():
                move = random.choice(current_state.get_valid_actions())
                current_state.make_move(move)

            winner = current_state.check_winner()
            while current_node is not None:
                current_node.visits += 1
                if winner == 2:
                    current_node.wins += 1
                elif winner == 1:
                    current_node.wins -= 1
                current_node = current_node.parent

        best_child = max(root.children, key=lambda n: n.visits)
        return root.state.get_valid_actions()[root.children.index(best_child)]

    def combined_action(self, env):
        """
        Decide the action for the AI (Player 2) using logical rules, MCTS, and DQN.
        """
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().squeeze()

        valid_actions = env.get_valid_actions()
        valid_q_values = {action: q_values[action - 1] for action in valid_actions}

        action = max(valid_q_values, key=lambda a: valid_q_values[a])
        max_q_value = valid_q_values[action]

        if max_q_value < 0.01:
            action = self.logic_based_action(env)
            if action is not None:
                logging.info(f"Logic-based action (win/block): {action}")
                return action
            action = self.monte_carlo_tree_search(env, num_simulations=100)
            if action is not None:
                logging.info(f"Using MCTS for action: {action}")
                return action
            return random.choice(valid_actions)

        logging.info(f"DQN action selected: Column {action}, Q-value: {max_q_value:.3f}")
        return action

    def calculate_reward(self, env, action, current_player):
        """
        Calculate the reward for Player 2 (agent) based on the game state and potential outcomes.
        """
        if env.check_winner() == 2:
            logging.info("Agent wins!")
            return 1.0
        elif env.check_winner() == 1:
            logging.info("Agent lost!")
            return -1.0
        elif env.is_draw():
            return 0.0

        if current_player == 2:
            if self.is_checkmate_move(env, action):
                return 5.0
            elif self.is_advantageous_move(env, action):
                return 0.5
            return -0.01
        else:
            return -0.05

    def is_checkmate_move(self, env, action):
        """
        Check if a move creates a checkmate opportunity for Player 2 (agent).
        """
        temp_env = deepcopy(env)
        temp_env.make_move(action)

        winning_moves = 0
        for col in temp_env.get_valid_actions():
            check_env = deepcopy(temp_env)
            check_env.make_move(col)
            if check_env.check_winner() == 2:
                winning_moves += 1
            if winning_moves >= 2:
                return True
        return False

    def is_advantageous_move(self, env, action):
        """
        Check if a move creates a winning opportunity for Player 2 (agent).
        """
        temp_env = deepcopy(env)
        temp_env.make_move(action)

        for col in temp_env.get_valid_actions():
            check_env = deepcopy(temp_env)
            check_env.make_move(col)
            if check_env.check_winner() == 2:
                return True
        return False
