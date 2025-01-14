from env.connect4 import Connect4
from models.dqn import DQN
from buffers.replay_buffer import DiskReplayBuffer
from agent.reward_system import RewardSystem
import random
import numpy as np
import threading
import multiprocessing as mp
from copy import deepcopy
import math
from agent.mcts import MCTS
class ActionSystem:
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def check_immediate_win(self, env, player):
        """Return the column that immediately wins for 'player', or None if none."""
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == player:
                return col
        return None

    def check_immediate_block(self, env, player):
        """Return the column that blocks an immediate win by the opponent, or None if none."""
        opponent = 3 - player
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == opponent:
                return col
        return None

    def pick_action(self, env, current_player, epsilon=0.1, episode=1):
        """
        This function tries:
          1) forced win
          2) forced block
          3) with probability epsilon -> random
          4) otherwise MCTS
          (You could also incorporate DQN’s Q-values if you like.)
        """
        # 1) Check forced win
        move = self.check_immediate_win(env, current_player)
        if move is not None:
            return move

        # 2) Check forced block
        move = self.check_immediate_block(env, current_player)
        if move is not None:
            return move

        # 3) Epsilon-random for exploration
        if random.random() < epsilon:
            valid_actions = env.get_valid_actions()
            if valid_actions:
                return random.choice(valid_actions)
            return None

        if episode < 50000:
            # Using logarithmic growth to adaptively scale simulations
            base_sims = 10  # Minimum number of simulations for small episodes
            scaling_factor = 0.1  # Adjust the growth rate
            sims = int(base_sims + scaling_factor * math.log1p(episode))  # log1p for stability
        else:
            sims=2500 # Best Performance

        # Ensure sims is at least 1
        sims = max(1, sims)
        num_processes = mp.cpu_count()
        # Initialize MCTS Agent for Evaluation
        mcts_evaluator = MCTSAgent(num_simulations=sims, num_processes=num_processes, debug=False)
        return mcts_evaluator.pick_action(env) 

    def pick_action_dqn(self, env):
        """
        Pure DQN pick action (greedy w.r.t. Q-values). 
        If you'd like \epsilon-greedy, you can incorporate that externally.
        """
        state_tensor = torch.tensor(env.board, dtype=torch.float32, device=self.policy_net.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        valid_actions = env.get_valid_actions()
        normalized_q_values = self.normalize_q_values(q_values)

        # Pick the action with the highest Q among valid actions
        valid_q = {a: normalized_q_values[a].item() for a in valid_actions}
        best_a = max(valid_q, key=lambda a: valid_q[a])
        return best_a

    def normalize_q_values(self, q_values):
        if isinstance(q_values, np.ndarray):
            q_values = torch.tensor(q_values, dtype=torch.float32)
        return torch.softmax(q_values, dim=0)
        
    def compute_reward(self, env, last_action, current_player):
        """
        External function that instantiates RewardSystem and calls its 
        'calculate_reward' method.
        
        Args:
            env: The current game environment.
            last_action: The action taken by the player.
            current_player: The player (1 or 2).
        
        Returns:
            tuple: (total_reward, win_status)
        """
        
        rs = RewardSystem()
        return rs.calculate_reward(env, last_action, current_player)
