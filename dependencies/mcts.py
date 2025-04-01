import math
import numpy as np
from copy import deepcopy
import logging
import random
import torch

from .environment import Connect4
from .utils import run_simulations_cuda 

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4

class MCTS:
    def __init__(self, logger=logging, num_simulations=4096, debug=False,
                 dqn_model=None, hybrid=False, q_threshold=0.5):
        self.num_simulations = num_simulations
        self.debug = debug
        self.logger = logger
        self.dqn_model = dqn_model  # Optional DQN to guide hybrid
        self.hybrid = hybrid        # Use DQN for hybrid if True
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
        state = np.expand_dims(state, axis=0)  # final shape: (1, 1, 6, 7)
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
        """
        Select the best action using MCTS with optional DQN integration.

        Returns:
            best_action (int): The column index chosen by MCTS (or fallback).
            mcts_value (float): A normalized value [0,1] computed as wins/total simulations.
            mcts_policy_dist (list of float): Probability-like distribution
                                              over all 7 columns (sum=1).
        """

        # 1) Immediate win
        move = self.check_immediate_win(env, current_player)
        if move is not None:
            if self.debug:
                self.logger.info(f"Immediate win by playing column {move}")
            policy_dist = [0.0]*COLUMNS
            policy_dist[move] = 1.0
            return move, 1.0, policy_dist

        # 2) Immediate block
        move = self.check_immediate_block(env, current_player)
        if move is not None:
            if self.debug:
                self.logger.info(f"Immediate block by playing column {move}")
            policy_dist = [0.0]*COLUMNS
            policy_dist[move] = 1.0
            return move, 0.5, policy_dist

        # 3) DQN-Guided Initialization: compute Q-values to create a bias array.
        #    We no longer return early here; we simply log the DQN suggestion and build a bias array.
        q_bias = np.zeros(COLUMNS, dtype=np.float32)
        valid_actions = env.get_valid_actions()
        if self.hybrid and self.dqn_model is not None:
            dqn_q_values = {}
            for action in valid_actions:
                temp_env = env.copy()
                temp_env.make_move(action)
                q_values_tensor = self.dqn_evaluate_state(temp_env)
                q_value = q_values_tensor.max().item()
                dqn_q_values[action] = q_value
                q_bias[action] = q_value  # use the Q-value as bias for that move

            best_action_dqn = max(dqn_q_values, key=dqn_q_values.get)
            best_q_value = dqn_q_values[best_action_dqn]
            if self.debug:
                self.logger.info(f"DQN suggests best action: {best_action_dqn} with Q-value: {best_q_value:.3f}")
        else:
            # If not using DQN-guided bias, leave q_bias as zeros (which will yield uniform randomness)
            pass

        # 4) Run MCTS Simulations on GPU, passing the q_bias to bias move selection.
        if self.debug:
            self.logger.info(f"Running MCTS with {self.num_simulations} simulations using CUDA.")

        if not valid_actions:
            return None, 0.0, [0.0]*COLUMNS

        simulation_results = run_simulations_cuda(env, self.num_simulations, q_bias=q_bias)
        if simulation_results is None:
            if self.debug:
                self.logger.error("Simulations failed. Returning random action.")
            ra = random.choice(valid_actions)
            policy_dist = [0.0]*COLUMNS
            policy_dist[ra] = 1.0
            return ra, 0.0, policy_dist

        # 5) Evaluate each action with sub-simulations
        simulations_per_action = self.num_simulations // len(valid_actions)
        extra_simulations = self.num_simulations % len(valid_actions)

        action_results = {}
        simulations_run = {}  # Track the number of simulations per action

        for action in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(action)
            sims = simulations_per_action + (1 if extra_simulations > 0 else 0)
            if extra_simulations > 0:
                extra_simulations -= 1

            simulations_run[action] = sims

            if sims <= 0:
                action_results[action] = 0.0
                continue

            # Run simulations for this action.
            simulation_outcomes = run_simulations_cuda(temp_env, sims, q_bias=q_bias)
            if simulation_outcomes is None:
                if self.debug:
                    self.logger.warning(f"Simulations for action {action} failed. Skipping.")
                action_results[action] = 0.0
                simulations_run[action] = 1  # Prevent division by zero later
                continue

            # Count only wins for the current_player (do not include draws)
            wins = np.sum(simulation_outcomes == current_player)
            action_results[action] = wins

        if self.debug:
            self.logger.debug(f"Aggregated action results (wins only): {action_results}")

        if not action_results:
            if self.debug:
                self.logger.warning("No valid action results after simulations. Choosing random action.")
            ra = random.choice(valid_actions)
            policy_dist = [0.0]*COLUMNS
            policy_dist[ra] = 1.0
            return ra, 0.0, policy_dist

        # 6) Pick the best action based on simulation win ratio
        best_action = max(action_results, key=action_results.get)
        best_score = action_results[best_action]
        total_sims_for_best = simulations_run.get(best_action, 1)

        if total_sims_for_best <= 0:
            mcts_value = 0.0
        else:
            mcts_value = best_score / total_sims_for_best
            mcts_value = min(max(mcts_value, 0.0), 1.0)

        # 7) Build MCTS policy distribution across all 7 columns
        sum_values = sum(action_results.values())
        if sum_values <= 0:
            mcts_policy_dist = [0.0]*COLUMNS
            uniform_prob = 1.0 / len(valid_actions)
            for a in valid_actions:
                mcts_policy_dist[a] = uniform_prob
        else:
            mcts_policy_dist = [0.0]*COLUMNS
            for a in range(COLUMNS):
                mcts_policy_dist[a] = action_results.get(a, 0.0) / sum_values

        # 8) Return best action, the normalized win ratio, and the distribution
        return best_action, mcts_value, mcts_policy_dist
