import os
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import logging
from numba.core.errors import NumbaPerformanceWarning
import warnings

from dependencies.environment import Connect4
from dependencies.models import DQN
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import setup_logger, safe_make_dir, get_next_index
from dependencies.mcts import MCTS

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Logging Setup ----------------- #

# Define the log directory and log file path
log_dir = os.path.join("data", "logs", "evaluation_logs")
log_file = os.path.join(log_dir, "manual_evaluation.log")

# Ensure the log directory exists before logging
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_file,  # Logs go here
    filemode="w",       # 'w' overwrites each run; use 'a' to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Logging setup complete!")

# ----------------- Model Paths ----------------- #
main_model_version = input("main_model_version>> ") or 2
print("main_model_version: ", main_model_version)
other_model_version = input("other_model_version>> ") or 1
print("other_model_version: ", other_model_version)
MODEL_PATH = os.path.join("data", "models", str(main_model_version), "Connect4_Agent_Model.pth")
OTHER_MODEL_PATH = os.path.join("data", "models", str(other_model_version), "Connect4_Agent_Model.pth")

# ----------------- Initialization ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- AutoEvaluator ----------------- #
class AutoEvaluator:
    """
    Class-based approach to keep 'self.env' as an attribute, instead of passing it around.
    """
    def __init__(self):
        # Use FixedConnect4 for piece-dropping logic
        self.env = Connect4()

    def play_one_game(self, agent1, agent2):
        """
        Plays a single game where agent1 (PLAYER1) starts and agent2 (PLAYER2) follows.
        Returns the winner: 1 (agent1), 2 (agent2), or 0 (draw).
        """
        self.env.reset()

        while True:
            # ---------------- AGENT 1 (Player 1) ----------------
            action_1 = agent1.pick_action(
                self.env,
                epsilon=0,
                logger=logging,
                debug=True,
                mcts_fallback=True,
                evaluation=False
            )

            if action_1 is None:  # No valid moves → draw
                return 0

            if isinstance(action_1, (tuple, list)):  # If pick_action returns a tuple, extract action
                action_1 = action_1[0]

            self.env.make_move(action_1)
            winner = self.env.check_winner()
            if winner != 0:
                return winner
            if self.env.is_draw():
                return 0

            # ---------------- AGENT 2 (Player 2) ----------------
            action_2 = agent2.pick_action(
                self.env,
                epsilon=0.1,
                logger=logging,
                debug=True,
                mcts_fallback=True,
                evaluation=True
            )

            if action_2 is None:
                return 0

            if isinstance(action_2, (tuple, list)):
                action_2 = action_2[0]

            self.env.make_move(action_2)
            winner = self.env.check_winner()
            if winner != 0:
                return winner
            if self.env.is_draw():
                return 0

    def evaluate_agents(self, agent1, agent2, n_episodes=100):
        """
        Plays agent1 vs agent2 for n_episodes.
        Returns (agent1_win_rate, agent2_win_rate, draw_rate).
        """
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for _ in range(n_episodes):
            winner = self.play_one_game(agent1, agent2)
            if winner == 1:
                agent1_wins += 1
            elif winner == 2:
                agent2_wins += 1
            else:
                draws += 1

        return (
            agent1_wins / n_episodes,
            agent2_wins / n_episodes,
            draws / n_episodes
        )

def main_evaluation(num_games=100):
    """
    Evaluates:
    1) DQN Model vs Random Opponent
    2) DQN Model vs MCTS Agent
    3) DQN Model vs Another DQN Model
    Also prints an overall win rate summary for the main agent.
    """
    print("\nStarting Model Evaluation")

    # Create an evaluator instance
    evaluator = AutoEvaluator()

    # Load main DQN Agent
    print(f"\nLoading Main DQN Agent from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    policy_net = DQN().to(device)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent_main = AgentLogic(policy_net, device)

    # Load the second DQN model
    checkpoint_other = torch.load(OTHER_MODEL_PATH, map_location=device)
    policy_net_other = DQN().to(device)
    policy_net_other.load_state_dict(checkpoint_other["policy_net_state_dict"])
    agent_other = AgentLogic(policy_net_other, device)

    # Variables to accumulate overall counts
    overall_main_wins = 0
    overall_opponent_wins = 0
    overall_draws = 0
    overall_games = 0

    # Evaluate vs Random Opponent
    print("\nEvaluating vs Random Opponent...")
    random_agent = AgentLogic(policy_net_other, device, mcts_simulations=0, always_random=True)
    a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=random_agent, agent2=agent_main, n_episodes=num_games)
    print(f"[VS Random] Random Win Rate = {a1_wr*100:.1f}% | Agent Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
    overall_opponent_wins += a1_wr * num_games
    overall_main_wins += a2_wr * num_games
    overall_draws += dr * num_games
    overall_games += num_games

    # Evaluate vs MCTS Agent with different simulation counts
    for sims in range(200, 2001, 200):
        mcts_agent = AgentLogic(policy_net_other, device, mcts_simulations=sims, always_mcts=True)
        print(f"\nEvaluating vs MCTS Agent ({sims} simulations)...")
        a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=mcts_agent, agent2=agent_main, n_episodes=num_games)
        print(f"[VS MCTS {sims}] MCTS Win Rate = {a1_wr*100:.1f}% | Agent Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
        overall_opponent_wins += a1_wr * num_games
        overall_main_wins += a2_wr * num_games
        overall_draws += dr * num_games
        overall_games += num_games

    # Evaluate vs Another DQN model
    print(f"\nEvaluating vs Another DQN Model: {OTHER_MODEL_PATH}")
    a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=agent_other, agent2=agent_main, n_episodes=num_games)
    print(f"[VS Other DQN] Agent1 Win Rate = {a1_wr*100:.1f}% | Agent2 Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
    overall_opponent_wins += a1_wr * num_games
    overall_main_wins += a2_wr * num_games
    overall_draws += dr * num_games
    overall_games += num_games

    # Calculate overall percentages
    overall_main_win_rate = overall_main_wins / overall_games
    overall_opponent_win_rate = overall_opponent_wins / overall_games
    overall_draw_rate = overall_draws / overall_games

    print("\nOverall Evaluation Summary:")
    print(f"Total Games: {overall_games}")
    print(f"Main Agent Overall Win Rate: {overall_main_win_rate*100:.1f}%")
    print(f"Opponents Overall Win Rate: {overall_opponent_win_rate*100:.1f}%")
    print(f"Overall Draw Rate: {overall_draw_rate*100:.1f}%")
    print("\nEvaluation Complete.")

# ------------------- Run Evaluation ------------------- #
if __name__ == "__main__":
    main_evaluation(num_games=10)