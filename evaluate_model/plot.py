import matplotlib.pyplot as plt
import re
import numpy as np
import os

# Get user input or use default values
version = input("model version>> ") or "1"

# Define the correct log file path inside data/logs/train_logs/
log_dir = os.path.join("data", "logs", "train_logs", version)
log_file_path = os.path.join(log_dir, "train.log")

# Ensure the directory exists before accessing the log file
os.makedirs(log_dir, exist_ok=True)

print("Log file path:", log_file_path)

total_episodes = int(input("total_episodes>> ") or 1000000)
print("Total Episodes:", total_episodes)
interval = int(input("interval>> ") or 1000)
print("Interval:", interval)
annotate_on = (input("annotate_on (true/false)>> ").strip().lower() or "false") == "true"
print("Annotate:", annotate_on)

def parse_log_file(log_file_path):
    """
    Reads the train.log file line by line and extracts:
      - winner
      - reward
      - turns
      - epsilon
      - mcts_level
      - mcts_count (per turn)
      - recalculation
    Returns lists for each metric as well as min_reward and max_reward observed.
    """
    winners = []
    rewards = []
    turns = []
    epsilons = []
    mcts_levels = []
    mcts_counts = []      # We'll store normalized MCTS usage (count / total_mcts_turns)
    min_reward = float('inf')
    max_reward = float('-inf')

    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Example line to match:
                # Episode 10: Winner=2,Win Rate=50.00%, Turn=15, Reward=1.0,
                # EPSILON=0.123, MCTS LEVEL=2, Cumulative MCTS used: 13/7, Recalculation: 1
                match = re.search(
                    r"Episode\s+\d+:"
                    r"\s+Winner=(-?\d+),Win Rate=[\d.]+%,"
                    r"\s+Turn=(\d+),\s+Reward=([-.\d]+),"
                    r"\s+EPSILON=([\d.e-]+),\s+MCTS LEVEL=(\d+),"
                    r"\s+Cumulative MCTS used:\s+(\d+)/(\d+),\s+Recalculation:\s+(\d+)",
                    line
                )
                if match:
                    winner = int(match.group(1))
                    turn = int(match.group(2))
                    reward = float(match.group(3))
                    epsilon = float(match.group(4))
                    mcts_level = int(match.group(5))
                    used_mcts = int(match.group(6))
                    total_mcts_turns = int(match.group(7))
                    recalculation = int(match.group(8))

                    winners.append(winner)
                    rewards.append(reward)
                    turns.append(turn)
                    epsilons.append(epsilon)
                    mcts_levels.append(mcts_level)

                    mcts_counts.append( used_mcts)

                    min_reward = min(min_reward, reward)
                    max_reward = max(max_reward, reward)

    except FileNotFoundError:
        print(f"Error: The log file at '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{log_file_path}': {e}")

    return (
        winners,
        rewards,
        turns,
        epsilons,
        mcts_levels,
        mcts_counts,
        min_reward,
        max_reward,
    )

def calculate_rate_of_change(values):
    """
    Returns a list of the changes (current - previous).
    The first element is None since there's no previous value for the first.
    """
    if not values:
        return []
    roc = [None]
    for i in range(1, len(values)):
        roc.append(values[i] - values[i - 1])
    return roc

def plot_data(
    winners,
    rewards,
    turns,
    epsilons,
    mcts_levels,
    mcts_counts,
    min_reward,
    max_reward,
    total_episodes=1000000,
    interval=1000,
    annotate_on=False,
):
    """
    Creates two separate figures:
      1) Figure 1 (interval-based metrics):
         - Avg Win Rate (P2, red), Avg Reward (lime), Avg Turns (cyan),
           Avg MCTS Usage (yellow), Avg Recalc (magenta)
      2) Figure 2 (per-game + interval-based Win Rate):
         - Epsilon (as %), MCTS Level, and Avg Win Rate (red)
    """
    total_games = len(winners)
    if total_games == 0:
        print("No data to plot.")
        return

    # ---- Gather interval-based metrics (including partial intervals) ----
    avg_winrates = []
    avg_rewards = []
    avg_turns = []
    avg_mcts_usage = []
    interval_x = []

    idx = 0
    while idx < total_games:
        end_idx = min(idx + interval, total_games)

        interval_winners = winners[idx:end_idx]
        interval_rewards = rewards[idx:end_idx]
        interval_turns = turns[idx:end_idx]
        interval_mcts_counts = mcts_counts[idx:end_idx]

        # 1) Interval-based P2 win rate
        if len(interval_winners) > 0:
            p2_wins = sum(1 for w in interval_winners if w == 2)
            interval_winrate_p2 = (p2_wins / len(interval_winners)) * 100.0
        else:
            interval_winrate_p2 = 0.0

        # 2) Interval-based average reward (ignoring zero reward)
        non_zero_rewards = [r for r in interval_rewards if r != 0]
        avg_reward = np.mean(non_zero_rewards) if non_zero_rewards else 0.0

        # 3) Interval-based average turns (ignoring zero turns)
        non_zero_turns = [t for t in interval_turns if t > 0]
        avg_turn = np.mean(non_zero_turns) if non_zero_turns else 0.0

        # 4) Average MCTS usage (already normalized)
        avg_mcts = np.mean(interval_mcts_counts) if interval_mcts_counts else 0.0


        avg_winrates.append(interval_winrate_p2)
        avg_rewards.append(avg_reward)
        avg_turns.append(avg_turn/2)
        avg_mcts_usage.append(avg_mcts)
        interval_x.append(end_idx)  # x for this chunk

        idx = end_idx

    if not avg_winrates:
        print("No intervals found.")
        return

    # ---------------------------------------------------------------------
    # PREPEND (0,0) so lines start at x=0
    # ---------------------------------------------------------------------
    interval_x.insert(0, 0)
    avg_winrates.insert(0, 0.0)
    avg_rewards.insert(0, 0.0)
    avg_turns.insert(0, 0.0)
    avg_mcts_usage.insert(0, 0.0)


    # ---- Rate-of-change (for optional annotation) ----
    roc_winrates = calculate_rate_of_change(avg_winrates)
    roc_rewards = calculate_rate_of_change(avg_rewards)
    roc_turns = calculate_rate_of_change(avg_turns)
    roc_mcts_usage = calculate_rate_of_change(avg_mcts_usage)
   

    # ---- Overall stats ----
    total_p2_wins = sum(1 for w in winners if w == 2)
    draws = winners.count(-1)
    last_game_idx = interval_x[-1]
    progress_percentage = (last_game_idx / total_episodes) * 100.0

    # ---------------------------------------------------------------------
    # FIGURE 1: Interval-based metrics only
    # ---------------------------------------------------------------------
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(figsize=(18, 8))

    line_winrate, = ax1.plot(
        interval_x,
        avg_winrates,
        label="Avg P2 WinRate (%)",
        linewidth=2,   
        color="green",   
    )
    line_reward, = ax1.plot(
        interval_x,
        avg_rewards,
        label="Avg Reward",
        linewidth=2,   
        color="Yellow",   
    )
    line_turns, = ax1.plot(
        interval_x,
        avg_turns,
        label="Avg Turns",
        linewidth=4,    
        color="cyan",  
    )
    line_mcts, = ax1.plot(
        interval_x,
        avg_mcts_usage,
        label="Avg MCTS Usage",
        linewidth=2,    
        color="red", 
    )
  

    # Optional annotations
    if annotate_on:
        for i, x_val in enumerate(interval_x):
            if i == 0:
                continue  # skip the dummy zero
            # WinRate
            if roc_winrates[i] is not None:
                plus_sign = "+" if roc_winrates[i] > 0 else ""
                y_val = avg_winrates[i]
                ax1.text(x_val, y_val, f"{plus_sign}{roc_winrates[i]:.2f}%", 
                         fontsize=7, ha="center", color="red")
            # Reward
            if roc_rewards[i] is not None:
                plus_sign = "+" if roc_rewards[i] > 0 else ""
                y_val = avg_rewards[i]
                ax1.text(x_val, y_val, f"{plus_sign}{roc_rewards[i]:.2f}", 
                         fontsize=7, ha="center", color="lime")
            # Turns
            if roc_turns[i] is not None:
                plus_sign = "+" if roc_turns[i] > 0 else ""
                y_val = avg_turns[i]
                ax1.text(x_val, y_val, f"{plus_sign}{roc_turns[i]:.2f}", 
                         fontsize=7, ha="center", color="cyan")
            # MCTS usage
            if roc_mcts_usage[i] is not None:
                plus_sign = "+" if roc_mcts_usage[i] > 0 else ""
                y_val = avg_mcts_usage[i]
                ax1.text(x_val, y_val, f"{plus_sign}{roc_mcts_usage[i]:.3f}", 
                         fontsize=7, ha="center", color="yellow")

    ax1.set_title(
        f"Figure 1: Interval-Based Metrics\n"
        f"Games Processed: {total_games}/{total_episodes} ({progress_percentage:.2f}% of target)\n"
        f"Overall P2 WinRate: {(total_p2_wins / total_games)*100:.2f}%, Draws: {draws}, "
        f"Min Reward: {min_reward:.2f}, Max Reward: {max_reward:.2f}"
    )
    ax1.set_xlabel("Game Index")
    ax1.set_ylabel("Interval-based Metrics")
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)

    lines_1 = [line_winrate, line_reward, line_turns, line_mcts]
    labels_1 = [l.get_label() for l in lines_1]
    ax1.legend(lines_1, labels_1, loc="upper left")

    fig1.tight_layout()

    # ---------------------------------------------------------------------
    # FIGURE 2: Epsilon(%) + MCTS Level (per game) + (interval-based) Avg Winrate
    # ---------------------------------------------------------------------
    fig2, ax2_left = plt.subplots(figsize=(18, 8))
    ax2_right = ax2_left.twinx()

    # Convert Epsilon to %
    epsilon_percent = [e * 100 for e in epsilons]
    x_games = np.arange(1, total_games + 1)

    # Epsilon line (left axis, lime)
    line_epsilon, = ax2_left.plot(
        x_games,
        epsilon_percent,
        label="Epsilon (%)",
        linewidth=0.5, 
        color="blue",
    )

    # Avg Win Rate line (RED), same data as figure 1
    line_avg_win, = ax2_left.plot(
        interval_x,
        avg_winrates,
        label="Avg WinRate (%) [interval]",
        linewidth=1,  
        color="green",
    )

    # MCTS Level line (right axis, yellow)
    line_mcts_level, = ax2_right.plot(
        x_games,
        mcts_levels,
        label="MCTS Level",
        linewidth=1,  
        color="white",
    )

    ax2_left.set_title(
        f"Figure 2: Epsilon(%) + MCTS Level + Avg Winrate\n"
        f"Games Processed: {total_games}/{total_episodes} ({progress_percentage:.2f}% of target)\n"
        f"Overall P2 WinRate: {(total_p2_wins / total_games)*100:.2f}%, Draws: {draws}"
    )
    ax2_left.set_xlabel("Game Index")
    ax2_left.set_ylabel("Epsilon (%) / Avg Winrate (%)")
    ax2_right.set_ylabel("MCTS Level")
    ax2_left.grid(color="gray", linestyle="--", linewidth=0.5)

    lines_left = [line_epsilon, line_avg_win]
    lines_right = [line_mcts_level]
    labels_left = [l.get_label() for l in lines_left]
    labels_right = [l.get_label() for l in lines_right]

    ax2_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    fig2.tight_layout()

    # ---------------------------------------------------------------------
    # Show both figures
    # ---------------------------------------------------------------------
    plt.show()

# ---- Main Execution ----
(
    winners,
    rewards,
    turns,
    epsilons,
    mcts_levels,
    mcts_counts,
    min_reward,
    max_reward,
) = parse_log_file(log_file_path)

plot_data(
    winners,
    rewards,
    turns,
    epsilons,
    mcts_levels,
    mcts_counts,
    min_reward,
    max_reward,
    total_episodes=total_episodes,
    interval=interval,
    annotate_on=annotate_on,
)
