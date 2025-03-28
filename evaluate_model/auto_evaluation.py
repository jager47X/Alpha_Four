import os
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import logging
import random
from numba.core.errors import NumbaPerformanceWarning
import warnings

from dependencies.environment import Connect4
from dependencies.agent import AgentLogic
from dependencies.replay_buffer import DiskReplayBuffer
from dependencies.utils import setup_logger, safe_make_dir, get_next_index
from dependencies.mcts import MCTS

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Logging Setup ----------------- #
log_dir = os.path.join("data", "logs", "evaluation_logs")
log_file = os.path.join(log_dir, "manual_evaluation.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_file,  
    filemode="w",       
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Logging setup complete!")

# ----------------- Model Paths ----------------- #
main_model_version = int(input("main_model_version>> ") or 2)
print("main_model_version: ", main_model_version)
other_model_version = int(input("other_model_version>> ") or 1)
print("other_model_version: ", other_model_version)
MODEL_PATH = os.path.join("data", "models", str(main_model_version), "Connect4_Agent_Model.pth")
OTHER_MODEL_PATH = os.path.join("data", "models", str(other_model_version), "Connect4_Agent_Model.pth")

if main_model_version >= 45:
    from dependencies.layer_models.model2 import DQN as main_DQN
else:
    from dependencies.layer_models.model1 import DQN as main_DQN

if other_model_version >= 45:
    from dependencies.layer_models.model2 import DQN as other_DQN
else:
    from dependencies.layer_models.model1 import DQN as other_DQN

# ----------------- AutoEvaluator Class ----------------- #
class AutoEvaluator:
    """
    Evaluator class that plays games between two agents and tracks main agent usage.
    """
    def __init__(self):
        self.env = Connect4()
        # Tracking main agent's (assumed agent2) usage.
        self.main_agent_usage = {"dqn": 0, "mcts": 0, "hybrid": 0, "random": 0, "total": 0}

    def play_one_game(self, agent1, agent2):
        """
        Plays one game between agent1 (Player 1) and agent2 (Player 2).
        Returns the winner: 1 (agent1), 2 (agent2), or 0 (draw).
        """
        self.env.reset()

        while True:
            # ---------------- AGENT 1 (Player 1) ----------------
            (_, q_action, mcts_action, hybrid_action,
             best_q_val, mcts_value, hybrid_value, random_action) = agent1.pick_action(
                self.env, epsilon=0, logger=logging, debug=True, mcts_fallback=True, hybrid=True
            )
            # For opponent (agent1) we don't track usage.
            if mcts_action is not None:
                action1 = mcts_action
            elif random_action is not None:
                action1 = random_action
            elif q_action is not None:
                action1 = q_action
            elif hybrid_action is not None:
                action1 = hybrid_action
            else:
                action1 = None

            if action1 is not None:
                if isinstance(action1, (tuple, list)):
                    action1 = action1[0]
                self.env.make_move(action1)

            winner = self.env.check_winner()
            if winner != 0:
                return winner
            if self.env.is_draw():
                return 0

            # ---------------- AGENT 2 (Player 2 - main agent) ----------------
            (model_used, q_action, mcts_action, hybrid_action,
             best_q_val, mcts_value, hybrid_value, random_action) = agent2.pick_action(
                self.env, epsilon=0, logger=logging, debug=True, mcts_fallback=True, hybrid=True
            )
            # Track main agent usage.
            self.main_agent_usage["total"] += 1
            if model_used in self.main_agent_usage:
                self.main_agent_usage[model_used] += 1

            if model_used == "mcts" and mcts_action is not None:
                action2 = mcts_action
            elif model_used == "random" and random_action is not None:
                action2 = random_action
            elif model_used == "dqn" and q_action is not None:
                action2 = q_action
            elif model_used == "hybrid" and hybrid_action is not None:
                action2 = hybrid_action
            else:
                action2 = None

            if action2 is not None:
                if isinstance(action2, (tuple, list)):
                    action2 = action2[0]
                self.env.make_move(action2)

            winner = self.env.check_winner()
            if winner != 0:
                return winner
            if self.env.is_draw():
                return 0

    def evaluate_agents(self, agent1, agent2, n_episodes=100):
        """
        Evaluates agent1 vs agent2 for n_episodes.
        Returns win rates as a tuple: (agent1_win_rate, agent2_win_rate, draw_rate).
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

# ----------------- Main Evaluation Function ----------------- #
def main_evaluation(num_games=100):
    """
    Evaluates three matchups:
      1) DQN Model vs Random Opponent
      2) DQN Model vs MCTS Agent (with varying simulations)
      3) DQN Model vs Another DQN Model
    Prints overall win rate and usage summary.
    """
    print("\nStarting Model Evaluation")
    evaluator = AutoEvaluator()

    # Load main DQN Agent
    print(f"\nLoading Main DQN Agent from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    policy_net = main_DQN().to(device)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent_main = AgentLogic(policy_net, device)

    # Load second DQN Agent
    print(f"\nLoading Second DQN Agent from: {OTHER_MODEL_PATH}")
    checkpoint_other = torch.load(OTHER_MODEL_PATH, map_location=device)
    policy_net_other = other_DQN().to(device)
    policy_net_other.load_state_dict(checkpoint_other["policy_net_state_dict"])
    agent_other = AgentLogic(policy_net_other, device)

    overall_main_wins = 0
    overall_opponent_wins = 0
    overall_draws = 0
    overall_games = 0

    # Evaluate vs Random Opponent
    print("\nEvaluating vs Random Opponent...")
    random_agent = AgentLogic(policy_net, device, q_threshold=0.9,temperature = 0.1, hybrid_value_threshold =1.0,mcts_simulations=2000, always_mcts=True, always_random=True)
    a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=random_agent, agent2=agent_main, n_episodes=num_games)
    print(f"[VS Random] Random Win Rate = {a1_wr*100:.1f}% | Agent Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
    overall_opponent_wins += a1_wr * num_games
    overall_main_wins += a2_wr * num_games
    overall_draws += dr * num_games
    overall_games += num_games

    # Evaluate vs MCTS Agent with different simulation counts
    for sims in range(200, 2001, 200):
        mcts_agent = AgentLogic( policy_net, device, q_threshold=0.9,temperature = 0.1, hybrid_value_threshold =1.0,mcts_simulations=2000, always_mcts=True, always_random=False)
        print(f"\nEvaluating vs MCTS Agent ({sims} simulations)...")
        a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=mcts_agent, agent2=agent_main, n_episodes=num_games)
        print(f"[VS MCTS {sims}] MCTS Win Rate = {a1_wr*100:.1f}% | Agent Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
        overall_opponent_wins += a1_wr * num_games
        overall_main_wins += a2_wr * num_games
        overall_draws += dr * num_games
        overall_games += num_games

    # Evaluate vs Another DQN Model
    print(f"\nEvaluating vs Another DQN Model: {OTHER_MODEL_PATH}")
    a1_wr, a2_wr, dr = evaluator.evaluate_agents(agent1=agent_other, agent2=agent_main, n_episodes=num_games)
    print(f"[VS Other DQN] Agent1 Win Rate = {a1_wr*100:.1f}% | Agent2 Win Rate = {a2_wr*100:.1f}% | Draw Rate = {dr*100:.1f}%")
    overall_opponent_wins += a1_wr * num_games
    overall_main_wins += a2_wr * num_games
    overall_draws += dr * num_games
    overall_games += num_games

    overall_main_win_rate = overall_main_wins / overall_games
    overall_opponent_win_rate = overall_opponent_wins / overall_games
    overall_draw_rate = overall_draws / overall_games

    # Usage summary for main agent (agent_main used as agent2 in every game)
    usage = evaluator.main_agent_usage
    total = usage["total"] if usage["total"] > 0 else 1  # Avoid division by zero
    dqn_pct = (usage["dqn"] / total) * 100
    mcts_pct = (usage["mcts"] / total) * 100
    hybrid_pct = (usage["hybrid"] / total) * 100
    random_pct = (usage["random"] / total) * 100
    hybrid_dqn_pct = ((usage["dqn"] + usage["hybrid"]) / total) * 100

    # Determine model maturity based on Hybrid+DQN usage.
    if hybrid_dqn_pct < 30:
        maturity = "immature"
    elif 50 <= hybrid_dqn_pct < 70:
        maturity = "developing"
    elif 80 <= hybrid_dqn_pct <= 100:
        maturity = "mature"
    else:
        maturity = "undetermined"

    # Determine performance comment based on win rate.
    if overall_main_win_rate < 0.3:
        performance = "low performance"
    elif overall_main_win_rate < 0.6:
        performance = "medium"
    elif overall_main_win_rate < 0.8:
        performance = "good"
    elif overall_main_win_rate < 0.9:
        performance = "impressive"
    elif overall_main_win_rate < 0.99:
        performance = "perfect"
    else:
        performance = "outstanding"

    # Print summary
    print("\nOverall Evaluation Summary:")
    print(f"Total Games: {overall_games}")
    print(f"Main Agent Overall Win Rate: {overall_main_win_rate*100:.1f}% ({performance})")
    print(f"Opponents Overall Win Rate: {overall_opponent_win_rate*100:.1f}%")
    print(f"Overall Draw Rate: {overall_draw_rate*100:.1f}%")
    print("\nMain Agent Action Usage:")
    print(f"  DQN: {dqn_pct:.1f}%")
    print(f"  MCTS: {mcts_pct:.1f}%")
    print(f"  Hybrid: {hybrid_pct:.1f}%")
    print(f"  Random: {random_pct:.1f}%")
    print(f"  (Hybrid + DQN): {hybrid_dqn_pct:.1f}% → Model is {maturity}.")
    print("\nEvaluation Complete.")

# ------------------- Run Evaluation ------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main_evaluation(num_games=10)
