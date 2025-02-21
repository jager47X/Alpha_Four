import matplotlib.pyplot as plt
import re
import numpy as np
import math  # Import math module for IQ calculation
# Get user input or use default values if input is empty
log_file_path = input("FILE PATH>> ") or "train.log"
print(log_file_path)
total_episodes = int(input("total_episodes>> ") or 2500000)
print(total_episodes)
interval = int(input("interval>> ") or 25000)
print(interval)
annotate_on = (input("annotate_on (true/false)>> ").strip().lower() or "true") == "true"
print(annotate_on)
# Function to parse the log file
def parse_log_file(log_file_path):
    rewards = []
    winners = []
    turns = []
    min_reward = float('inf')
    max_reward = float('-inf')

    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Use an optional group for the Turn value
                match = re.search(r"Winner=(-?\d+)(?:, Turn=(\d+))?, Reward=([-.\d]+)", line)
                if match:
                    winner = int(match.group(1))
                    turn = int(match.group(2)) if match.group(2) is not None else 0
                    reward = float(match.group(3))
                    
                    # Update min and max rewards
                    if reward > max_reward:
                        max_reward = reward
                    if reward < min_reward:
                        min_reward = reward

                    winners.append(winner)
                    rewards.append(reward)
                    turns.append(turn)
    except FileNotFoundError:
        print(f"Error: The log file at '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return winners, rewards, turns, min_reward, max_reward

# Function to calculate rate of change
def calculate_rate_of_change(metric_list):
    """
    Calculate the rate of change between consecutive elements in a list.
    Returns a list where each element is the change from the previous element.
    The first element is set to None as there is no previous element.
    """
    rate_of_change = [None]  # First element has no previous element
    for i in range(1, len(metric_list)):
        change = metric_list[i] - metric_list[i - 1]
        rate_of_change.append(change)
    return rate_of_change

# Function to calculate IQ
def calculate_iq(w, r, s):
    """
    Calculate the IQ metric based on the provided formula.
    IQ = e^w * sqrt(r) * ln(s)
    
    Parameters:
    - w (float): Total average Player 2 win rate [0.0, 1.0]
    - r (float): Average reward in the interval [1, 42]
    - s (int): Current episode number (>= 2)
    
    Returns:
    - iq (float): Calculated IQ metric
    """
    try:
        # Compute the result
        if r <= 0:  # Ensure r is >0
            print(f"Average Reward is negative or zero: {r}. Setting to 0.0001 instead.")
            r = 0.0001
        result = math.exp(w) * math.sqrt(r / 2) * math.log(s)
        return result
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        print(f"Error calculating IQ: {e}")
        return 0  # Handle any mathematical errors gracefully

# Function to plot the data
def plot_data(winners, rewards, turns, min_reward, max_reward, total_episodes=100000, interval=1000, annotate_on=False):
    if not rewards or not winners and not turns:
        print("No data to plot.")
        return

    # Calculate total games
    total_games = len(winners)
    
    # Calculate winner counts
    winner_counts = {
        1: winners.count(1),   # Wins for Player 1
        2: winners.count(2),   # Wins for Player 2
        -1: winners.count(-1)  # Draws
    }

    # Compute cumulative win rates
    cumulative_games = np.arange(1, total_games + 1)  # Game indices
    cumulative_wins_player1 = np.cumsum(np.array(winners) == 1)
    cumulative_wins_player2 = np.cumsum(np.array(winners) == 2)

    # Ensure cumulative_wins_player1 and cumulative_wins_player2 are floats
    cumulative_wins_player1 = cumulative_wins_player1.astype(float)
    cumulative_wins_player2 = cumulative_wins_player2.astype(float)

    # Calculate win rates as percentages
    win_rate_player1 = (cumulative_wins_player1 / cumulative_games) * 100  # 0 to 100
    win_rate_player2 = (cumulative_wins_player2 / cumulative_games) * 100  # 0 to 100

    # Compute average rewards for each complete interval, excluding rewards of 0.0
    avg_rewards = []
    avg_rewards_x = []
    for i, start in enumerate(range(0, len(rewards), interval)):
        end = start + interval
        interval_rewards = rewards[start:end]
        
        # Process only complete intervals
        if len(interval_rewards) != interval:
            print(f"Skipping incomplete interval {i+1}: Games {start+1} to {end} (only {len(interval_rewards)} games)")
            continue
        
        # Exclude rewards that are 0.0
        non_zero_rewards = [r for r in interval_rewards if r != 0.0]
        
        # Calculate the average of non-zero rewards
        if non_zero_rewards:
            avg = np.mean(non_zero_rewards)
        else:
            avg = 0.0  # or you can choose to set it to `np.nan` if preferred
        
        avg_rewards.append(avg)
        avg_rewards_x.append(end)

    # Compute average turns for each complete interval
    avg_turns = []
    avg_turns_x = []
    for i, start in enumerate(range(0, len(turns), interval)):
        end = start + interval
        interval_turns = turns[start:end]
        
        # Process only complete intervals
        if len(interval_turns) != interval:
            print(f"Skipping incomplete interval {i+1}: Games {start+1} to {end} (only {len(interval_turns)} games)")
            continue
        
        # Exclude turns that are 0
        non_zero_turns = [t for t in interval_turns if t > 0]
        
        # Calculate the average of non-zero turns
        if non_zero_turns:
            avg = np.mean(non_zero_turns)
        else:
            avg = 0.0  # or you can choose to set it to `np.nan` if preferred
        
        avg_turns.append(avg)
        avg_turns_x.append(end)

    # Compute IQ values for each complete interval using w_total
    # First, calculate total average win rate for Player 2 (w_total)
    if total_games == 0:
        w_total = 0.0
    else:
        w_total = cumulative_wins_player2[-1] / cumulative_games[-1]  # Total average P2 win rate
    w_total = float(w_total)  # Ensure float
    w_total = min(max(w_total, 0.0), 1.0)  # Clamp to [0.0, 1.0]

    print(f"Total average Player 2 win rate (w_total): {w_total}")

    iq_values = []
    iq_x = []
    for i, start in enumerate(range(0, total_games, interval)):
        end = start + interval
        interval_rewards = rewards[start:end]
        interval_turns = turns[start:end]
        
        # Process only complete intervals
        if len(interval_rewards) != interval:
            continue  # Skip incomplete interval

        # Exclude rewards that are 0.0
        non_zero_rewards = [r for r in interval_rewards if r != 0.0]

        # Calculate average reward (r)
        r = np.mean(non_zero_rewards) if non_zero_rewards else 0.0  # Default to 0.0 if no non-zero rewards

        # Exclude turns that are 0
        valid_interval_turns = [turn for turn in interval_turns if turn > 0]
        t = np.mean(valid_interval_turns) if valid_interval_turns else 1  # Default to 1

        # Use w_total as 'w' for IQ calculation
        w = w_total  # Total average P2 win rate

        # Current episode (s) is the end of the interval
        s = max(end, 2)  # Ensure s > 1 for log(s)

        # Debugging: Print intermediate values (Optional)
        print(f"Interval {i+1}:")
        print(f"  r (avg_reward) = {r}")
        print(f"  t (avg_turns) = {t}")
        print(f"  w (total_average_win_rate) = {w}")
        print(f"  s (episode) = {s}")

        # Calculate IQ using the separate function
        iq = calculate_iq(w, r, s)
        print(f"  iq = {iq}\n")  # Expected: ~59.8984 for sample inputs

        iq_values.append(iq)
        iq_x.append(end)

    # Initialize plots with 0 at episode 0 (only if there is at least one complete interval)
    if avg_rewards and avg_turns and iq_values:
        avg_rewards = [0] + avg_rewards
        avg_rewards_x = [0] + avg_rewards_x

        avg_turns = [0] + avg_turns
        avg_turns_x = [0] + avg_turns_x

        iq_values = [0] + iq_values
        iq_x = [0] + iq_x
    else:
        print("No complete intervals to plot.")
        return

    # Calculate rate of change for all metrics
    roc_avg_rewards = calculate_rate_of_change(avg_rewards)
    roc_avg_turns = calculate_rate_of_change(avg_turns)
    roc_iq_values = calculate_rate_of_change(iq_values)

    # Compute win rate changes for annotations every interval
    rate_change_player1 = []
    rate_change_player2 = []
    annotation_x = []
    for i, start in enumerate(range(0, total_games, interval)):
        end = start + interval
        if len(rewards[start:end]) != interval:
            continue  # Skip incomplete interval
        # Ensure that we don't go out of bounds
        if end - 1 >= total_games:
            continue
        # Calculate change
        change_p1 = win_rate_player1[end - 1] - win_rate_player1[start]
        change_p2 = win_rate_player2[end - 1] - win_rate_player2[start]
        rate_change_player1.append(change_p1)
        rate_change_player2.append(change_p2)
        annotation_x.append(end)

    # Calculate progress percentage based on complete intervals
    completed_intervals = len(avg_rewards) - 1  # Exclude the initial 0
    progress_percentage = (completed_intervals * interval / total_episodes) * 100

    # Set black background style
    plt.style.use('dark_background')

    # Create figure and axis
    plt.figure(figsize=(18, 12))

    # Plot win rates
    #plt.plot(cumulative_games, win_rate_player1, label="Trainer (P1) Win Rate (%)", color="cyan", linewidth=1)
    #plt.plot(cumulative_games, win_rate_player2, label="Agent (P2) Win Rate (%)", color="orange", linewidth=1)

    # Plot average rewards
    plt.plot(avg_rewards_x, avg_rewards, label=f"Average Reward", color="lime", linewidth=2)

    # Plot average turns if available
    if any(turn > 0 for turn in turns):
        plt.plot(avg_turns_x, avg_turns, label=f"Average Game Length", color="magenta", linewidth=2)

    # Plot IQ values
    #if iq_values:
      #  plt.plot(iq_x, iq_values, label="Agent IQ Metric", color="yellow", linewidth=2)

    # *** Begin: Compute and Plot Per-Interval P2 Win Rate ***
    # Initialize list to store per-interval P2 win rates
    per_interval_p2_win_rates = []
    per_interval_p2_win_rates_x = []

    for i in range(completed_intervals):
        start = i * interval
        end = (i + 1) * interval
        interval_wins_p2 = winners[start:end].count(2)
        p2_win_rate = (interval_wins_p2 / interval) * 100  # Percentage
        per_interval_p2_win_rates.append(p2_win_rate)
        per_interval_p2_win_rates_x.append(end)

    # Plot per-interval P2 win rate
    plt.plot(per_interval_p2_win_rates_x, per_interval_p2_win_rates, label="Agent (P2) Win Rate per Interval (%)", color="blue", linewidth=2)
    # *** End: Compute and Plot Per-Interval P2 Win Rate ***

    if annotate_on:
        # Annotate rate changes for Player 1 win rate
        #for idx, (x, change) in enumerate(zip(annotation_x, rate_change_player1)):
          #  if change is not None and x != 0:
           #     plus_sign = "+" if change > 0 else ""
           ##     y_offset = win_rate_player1[x - 1] + 2  # Offset for annotation
            #    plt.text(
            #        x, y_offset, f"{plus_sign}{change:.2f}%", color="cyan", fontsize=8, ha="center"
             #   )

        # Annotate rate changes for Player 2 win rate
      #  for idx, (x, change) in enumerate(zip(annotation_x, rate_change_player2)):
            #if change is not None and x != 0:
              #  plus_sign = "+" if change > 0 else ""
              #  y_offset = win_rate_player2[x - 1] - 2  # Offset for annotation
             #   plt.text(
              #      x, y_offset, f"{plus_sign}{change:.2f}%", color="orange", fontsize=8, ha="center"
             #   )

        # Annotate rate changes for Average Rewards
        for i, (x, change) in enumerate(zip(avg_rewards_x, roc_avg_rewards)):
            if change is not None and x != 0:
                plus_sign = "+" if change > 0 else ""
                # Prevent division by zero in y_offset calculation
                y_offset = avg_rewards[i] + (0.05 * avg_rewards[i]) if avg_rewards[i] != 0 else avg_rewards[i] + 1
                plt.text(
                    x, y_offset, f"{plus_sign}{change:.2f}", color="lime", fontsize=8, ha="center"
                )

        # Annotate rate changes for Average Turns
        for i, (x, change) in enumerate(zip(avg_turns_x, roc_avg_turns)):
            if change is not None and x != 0 and avg_turns[i] != 0:
                plus_sign = "+" if change > 0 else ""
                # Prevent division by zero in y_offset calculation
                y_offset = avg_turns[i] + (0.05 * avg_turns[i]) if avg_turns[i] != 0 else avg_turns[i] + 1
                plt.text(
                    x, y_offset, f"{plus_sign}{change:.2f}", color="magenta", fontsize=8, ha="center"
                )

        # Annotate rate changes for IQ Metric
        #for i, (x, change) in enumerate(zip(iq_x, roc_iq_values)):
            #if change is not None and x != 0:
             #   plus_sign = "+" if change > 0 else ""
                # Prevent division by zero in y_offset calculation
              #  y_offset = iq_values[i] + (0.05 * iq_values[i]) if iq_values[i] != 0 else iq_values[i] + 1
              #  plt.text(
              #      x, y_offset, f"{plus_sign}{change:.3f}", color="yellow", fontsize=8, ha="center"
              #  )

        # *** Begin: Annotate Per-Interval P2 Win Rate ***
        for i, (x, p2_win_rate) in enumerate(zip(per_interval_p2_win_rates_x, per_interval_p2_win_rates)):
            if p2_win_rate is not None and x != 0:
                # Calculate the rate of change for P2 win rate
                if i == 0:
                    change = p2_win_rate  # First interval, change is the win rate itself
                else:
                    change = p2_win_rate - per_interval_p2_win_rates[i - 1]
                plus_sign = "+" if change > 0 else ""
                plt.text(
                    x, p2_win_rate + 1, f"{plus_sign}{change:.2f}%", color="blue", fontsize=8, ha="center"
                )
        # *** End: Annotate Per-Interval P2 Win Rate ***

    # Display winner statistics and progress percentage in the title
    plt.title(
        f"Win Rates, Average Rewards, Average Turns, IQ, and P2 Win Rate Over Time\n"
        f"Total Games: {total_games}/{total_episodes} ({progress_percentage:.2f}% Completed)\n"
        f"Agent Winrate: {(winner_counts.get(2, 0)/total_games)*100:.2f}%, "
        f"Draws: {winner_counts.get(-1, 0)}, "
        f"Agent MIN Reward: {min_reward:.2f}, "
        f"Agent MAX Reward: {max_reward:.2f}"
    )
    plt.xlabel("Game Index")
    plt.ylabel("Percentage / Reward / Turns / IQ")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Parse the log file and plot the data
winners, rewards, turns, min_reward, max_reward = parse_log_file(log_file_path)
plot_data(winners, rewards, turns, min_reward, max_reward, total_episodes, interval, annotate_on)
