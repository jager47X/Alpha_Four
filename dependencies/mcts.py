import math
import numpy as np
from copy import deepcopy
import logging
import random
import torch

from .environment import Connect4
from .utils import run_batched_simulations_cuda, run_batched_simulations_cpu

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4


def _run_batched_simulations(envs, sims_per_env, q_bias=None):
    """Try batched CUDA first, fall back to batched CPU."""
    try:
        result = run_batched_simulations_cuda(envs, sims_per_env, q_bias=q_bias)
        if result is not None:
            return result
    except Exception:
        pass
    return run_batched_simulations_cpu(envs, sims_per_env, q_bias=q_bias)


class MCTS:
    def __init__(self, logger=logging, num_simulations=4096, debug=False,
                 dqn_model=None, hybrid=False, q_threshold=0.5):
        self.num_simulations = num_simulations
        self.debug = debug
        self.logger = logger
        self.dqn_model = dqn_model  # Optional DQN to guide hybrid
        self.hybrid = hybrid        # Use DQN for hybrid if True
        self.q_threshold = q_threshold

    def dqn_evaluate_batch(self, envs):
        """
        Evaluate multiple board states in a single batched DQN forward pass.

        Args:
            envs: list of Connect4 environments to evaluate.

        Returns:
            torch.Tensor of shape (len(envs), num_actions) with Q-values.
        """
        states = []
        for env in envs:
            state = np.array(env.get_state(), dtype=np.float32)
            if state.ndim == 2:
                state = np.expand_dims(state, axis=0)
            states.append(state)
        batch = np.stack(states, axis=0)  # shape: (N, 1, 6, 7)
        batch_tensor = torch.from_numpy(batch)
        device = next(self.dqn_model.parameters()).device
        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            q_values = self.dqn_model(batch_tensor)
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

        # 3) Build per-action environments
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, 0.0, [0.0]*COLUMNS

        action_envs = []
        for action in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(action)
            action_envs.append(temp_env)

        # 4) DQN-Guided bias (batched single forward pass)
        q_bias = np.zeros(COLUMNS, dtype=np.float32)
        if self.hybrid and self.dqn_model is not None:
            q_values_batch = self.dqn_evaluate_batch(action_envs)
            dqn_q_values = {}
            for i, action in enumerate(valid_actions):
                q_value = q_values_batch[i].max().item()
                dqn_q_values[action] = q_value
                q_bias[action] = q_value

            best_action_dqn = max(dqn_q_values, key=dqn_q_values.get)
            best_q_value = dqn_q_values[best_action_dqn]
            if self.debug:
                self.logger.info(f"DQN suggests best action: {best_action_dqn} with Q-value: {best_q_value:.3f}")

        # 5) Run all MCTS simulations in a single batched kernel launch
        if self.debug:
            self.logger.info(f"Running MCTS with {self.num_simulations} simulations (batched).")

        simulations_per_action = self.num_simulations // len(valid_actions)
        extra_simulations = self.num_simulations % len(valid_actions)

        sims_per_env = []
        simulations_run = {}
        for i, action in enumerate(valid_actions):
            sims = simulations_per_action + (1 if i < extra_simulations else 0)
            sims_per_env.append(sims)
            simulations_run[action] = sims

        batch_results = _run_batched_simulations(action_envs, sims_per_env, q_bias=q_bias)
        if batch_results is None:
            if self.debug:
                self.logger.error("Batched simulations failed. Returning random action.")
            ra = random.choice(valid_actions)
            policy_dist = [0.0]*COLUMNS
            policy_dist[ra] = 1.0
            return ra, 0.0, policy_dist

        # 6) Count wins per action
        action_results = {}
        for i, action in enumerate(valid_actions):
            outcomes = batch_results[i]
            wins = np.sum(outcomes == current_player)
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

        # 7) Pick the best action based on simulation win ratio
        best_action = max(action_results, key=action_results.get)
        best_score = action_results[best_action]
        total_sims_for_best = simulations_run.get(best_action, 1)

        if total_sims_for_best <= 0:
            mcts_value = 0.0
        else:
            mcts_value = best_score / total_sims_for_best
            mcts_value = min(max(mcts_value, 0.0), 1.0)

        # 8) Build MCTS policy distribution across all 7 columns
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

        # 9) Return best action, the normalized win ratio, and the distribution
        return best_action, mcts_value, mcts_policy_dist
