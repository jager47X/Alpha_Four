import math
import numpy as np
from copy import deepcopy
import logging
import random
import torch

from .environment import Connect4
from .utils import run_simulations_cuda  # Assuming this function exists

# Constants
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4

class MCTS:
    def __init__(self, logger=logging, num_simulations=4096, debug=False, dqn_model=None, evaluation=False, q_threshold=0.5):
        self.num_simulations = num_simulations
        self.debug = debug
        self.logger = logger
        self.dqn_model = dqn_model  # Optional DQN to guide evaluation
        self.evaluation = evaluation  # Use DQN for evaluation if True
        self.q_threshold = q_threshold

    def dqn_evaluate_state(self, env):
        """
        Evaluate the current state using the DQN model.
        Assumes that env.get_state() returns a 2D board of shape (ROWS, COLUMNS).
        The state is then reshaped to (1, 1, ROWS, COLUMNS) as expected by a CNN.
        """
        state = np.array(env.get_state(), dtype=np.float32)
        if state.ndim == 2:
            state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)
        state_tensor = torch.from_numpy(state)
        device = next(self.dqn_model.parameters()).device
        state_tensor = state_tensor.to(device)
        q_values = self.dqn_model(state_tensor)
        return q_values

    def check_immediate_win(self, env, player):
        """Check if an immediate winning move is available."""
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == player:
                return col
        return None

    def check_immediate_block(self, env, player):
        """Check if an immediate block is required."""
        opponent = 3 - player
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == opponent:
                return col
        return None

    def select_action(self, env, current_player):
        """Select the best action using MCTS with optional DQN integration."""

        # 1) Immediate win
        move = self.check_immediate_win(env, current_player)
        if move is not None:
            if self.debug:
                self.logger.info(f"Immediate win by playing column {move}")
            return move, 1.0

        # 2) Immediate block
        move = self.check_immediate_block(env, current_player)
        if move is not None:
            if self.debug:
                self.logger.info(f"Immediate block by playing column {move}")
            return move, 0.5

        # 3) DQN-Guided Initialization (only in evaluation mode)
        if self.evaluation and self.dqn_model is not None:
            valid_actions = env.get_valid_actions()
            dqn_q_values = {}
            for action in valid_actions:
                temp_env = env.copy()
                temp_env.make_move(action)
                q_values_tensor = self.dqn_evaluate_state(temp_env)
                q_value = q_values_tensor.max().item()
                dqn_q_values[action] = q_value

            best_action = max(dqn_q_values, key=dqn_q_values.get)
            best_q_value = dqn_q_values[best_action]
            if best_q_value >= self.q_threshold:
                if self.debug:
                    self.logger.info(f"DQN suggests best action: {best_action} with Q-value: {best_q_value:.3f}")
                return best_action, best_q_value
            else:
                if self.debug:
                    self.logger.info(f"DQN best Q-value {best_q_value:.3f} below threshold {self.q_threshold}. Falling back to MCTS simulations.")

        # 4) Run MCTS Simulations on GPU
        if self.debug:
            self.logger.info(f"Running MCTS with {self.num_simulations} simulations using CUDA.")

        simulation_results = run_simulations_cuda(env, self.num_simulations)
        if simulation_results is None:
            if self.debug:
                self.logger.error("Simulations failed. Returning random action.")
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                return None, 0.0
            return random.choice(valid_actions), 0.0

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, 0.0

        # 5) Assign Simulations per Action, with DQN leaf evaluation if confident,
        #    otherwise fall back to simulation outcomes.
        simulations_per_action = self.num_simulations // len(valid_actions)
        extra_simulations = self.num_simulations % len(valid_actions)

        action_results = {}
        for action in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(action)
            sims = simulations_per_action + (1 if extra_simulations > 0 else 0)
            if extra_simulations > 0:
                extra_simulations -= 1
            if sims == 0:
                continue

            if self.evaluation and self.dqn_model is not None:
                q_values_tensor = self.dqn_evaluate_state(temp_env)
                value_estimate = q_values_tensor.max().item()
                if value_estimate >= self.q_threshold:
                    # Use the DQN value if confident
                    action_results[action] = value_estimate
                else:
                    # Otherwise, fall back to standard MCTS simulation for this leaf.
                    simulation_outcomes = run_simulations_cuda(temp_env, sims)
                    if simulation_outcomes is None:
                        if self.debug:
                            self.logger.warning(f"Simulations for action {action} failed. Skipping.")
                        continue
                    wins = np.sum(simulation_outcomes == (PLAYER1 if current_player == PLAYER1 else PLAYER2))
                    draws = np.sum(simulation_outcomes == EMPTY)
                    action_results[action] = wins + 0.5 * draws
            else:
                simulation_outcomes = run_simulations_cuda(temp_env, sims)
                if simulation_outcomes is None:
                    if self.debug:
                        self.logger.warning(f"Simulations for action {action} failed. Skipping.")
                    continue
                wins = np.sum(simulation_outcomes == (PLAYER1 if current_player == PLAYER1 else PLAYER2))
                draws = np.sum(simulation_outcomes == EMPTY)
                action_results[action] = wins + 0.5 * draws

        if self.debug:
            self.logger.debug(f"Aggregated action results: {action_results}")

        if not action_results:
            if self.debug:
                self.logger.warning("No valid action results after simulations. Choosing random action.")
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                return None, 0.0
            return random.choice(valid_actions), 0.0

        # 6) Select the Best Action from MCTS simulations
        best_action = max(action_results, key=action_results.get)
        best_value = action_results[best_action]

        if self.debug:
            self.logger.debug(f"Chose best action {best_action} with aggregated value {best_value}")

        return best_action, best_value
