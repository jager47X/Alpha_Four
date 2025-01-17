import matplotlib.pyplot as plt
import re
import numpy as np
import math  # Import math module for IQ calculation

# Path to the log file
log_file_path = input("FILE PATH>> ")

# Function to parse the log file
def parse_log_file(log_file_path):
    rewards = []
    winners = []
    turns = []
    min_reward = float('inf')  # Initialize to a large value
    max_reward = float('-inf')  # Initialize to a small value

    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Attempt to extract Winner, Turn, and Reward
                match = re.search(r"Winner=(-?\d+),Turn=(\d+), Reward=([-.\d]+)", line)
                if match:
                    winner = int(match.group(1))  # Winner (1, 2, or -1 for draws)
                    turn = int(match.group(2))    # Number of turns
                    reward = float(match.group(3))  # Reward

                    # Update min and max rewards
                    if reward > max_reward:
                        max_reward = reward
                    if reward < min_reward:
                        min_reward = reward

                    # Append the extracted values
                    winners.append(winner)
                    rewards.append(reward)
                    turns.append(turn)
                else:
                    # Attempt to extract Winner and Reward without Turn
                    match = re.search(r"Winner=(-?\d+), Reward=([-.\d]+)", line)
                    if match:
                        winner = int(match.group(1))  # Winner (1, 2, or -1 for draws)
                        reward = float(match.group(2))  # Reward
                        turn = 0  # Assuming 0 when Turn is missing

                        # Update min and max rewards
                        if reward > max_reward:
                            max_reward = reward
                        if reward < min_reward:
                            min_reward = reward

                        # Append the extracted values
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
        return math.exp(w) * math.sqrt(r) * math.log(s)
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        print(f"Error calculating IQ: {e}")
        return 0  # Handle any mathematical errors gracefully

# Function to plot the data
def plot_data(winners, rewards, turns, min_reward, max_reward, total_episodes=100000, interval=1000):
    if not rewards or not winners or not turns:
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

    # Compute average rewards for each interval
    avg_rewards = [
        np.mean(rewards[start:start + interval])
        for start in range(0, len(rewards), interval)
        if len(rewards[start:start + interval]) == interval  # Ensure full interval
    ]
    avg_rewards_x = list(range(interval, interval * len(avg_rewards) + 1, interval))

    # Compute average turns for each interval
    avg_turns = [
        np.mean([turn for turn in turns[start:start + interval] if turn > 0]) if any(turn > 0 for turn in turns[start:start + interval]) else 0
        for start in range(0, len(turns), interval)
        if len(turns[start:start + interval]) == interval  # Ensure full interval
    ]
    avg_turns_x = list(range(interval, interval * len(avg_turns) + 1, interval))

    # Compute win rate changes for annotations every interval
    rate_change_player1 = [
        win_rate_player1[min(start + interval - 1, total_games - 1)] - win_rate_player1[start]
        for start in range(0, total_games, interval)
        if start + interval <= total_games  # Ensure full interval
    ]
    rate_change_player2 = [
        win_rate_player2[min(start + interval - 1, total_games - 1)] - win_rate_player2[start]
        for start in range(0, total_games, interval)
        if start + interval <= total_games  # Ensure full interval
    ]
    annotation_x = list(range(interval, interval * len(rate_change_player1) + 1, interval))

    # Calculate progress percentage
    progress_percentage = (total_games / total_episodes) * 100

    # Calculate total average win rate for Player 2 (w_total)
    if total_games == 0:
        w_total = 0.0
    else:
        w_total = cumulative_wins_player2[-1] / cumulative_games[-1]  # Total average P2 win rate
    w_total = float(w_total)  # Ensure float
    w_total = min(max(w_total, 0.0), 1.0)  # Clamp to [0.0, 1.0]

    print(f"Total average Player 2 win rate (w_total): {w_total}")

    # Calculate IQ values for each interval using w_total
    iq_values = []
    iq_x = []

    for i, start in enumerate(range(0, total_games, interval)):
        end = start + interval
        if end > total_games:
            break  # Avoid index out of range

        # Extract interval data
        interval_rewards = rewards[start:end]
        interval_turns = turns[start:end]
        interval_winners = winners[start:end]

        # Calculate average reward (r)
        r = np.mean(interval_rewards) if interval_rewards else 1  # Default to 1 if empty
        r = min(max(r, 1), 42)  # Clamp r to [1, 42]

        # Calculate average turns (t)
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

    # Initialize plots with 0 at episode 0
    avg_rewards = [0] + avg_rewards
    avg_rewards_x = [0] + avg_rewards_x

    avg_turns = [0] + avg_turns
    avg_turns_x = [0] + avg_turns_x

    iq_values = [0] + iq_values
    iq_x = [0] + iq_x

    # Calculate rate of change for all metrics
    roc_avg_rewards = calculate_rate_of_change(avg_rewards)
    roc_avg_turns = calculate_rate_of_change(avg_turns)
    roc_iq_values = calculate_rate_of_change(iq_values)

    # Adjust annotation_x to include 0 if necessary
    annotation_x = [0] + annotation_x

    # Set black background style
    plt.style.use('dark_background')

    # Create figure and axis
    plt.figure(figsize=(18, 12))

    # Plot win rates
    plt.plot(cumulative_games, win_rate_player1, label="Trainer (P1) Win Rate (%)", color="cyan", linewidth=1)
    plt.plot(cumulative_games, win_rate_player2, label="Agent (P2) Win Rate (%)", color="orange", linewidth=1)

    # Plot average rewards
    plt.plot(avg_rewards_x, avg_rewards, label=f"Average Reward", color="lime", linewidth=2)

    # Plot average turns if available
    if any(turn > 0 for turn in turns):
        plt.plot(avg_turns_x, avg_turns, label=f"Average Game Length", color="magenta", linewidth=2)

    # Plot IQ values
    if iq_values:
        plt.plot(iq_x, iq_values, label="Agent IQ Metric", color="yellow", linewidth=2)

    # Annotate rate changes for Player 1 win rate
    for idx, (x, change) in enumerate(zip(annotation_x, [None] + rate_change_player1)):
        if change is not None and x != 0:
            plus_sign = "+" if change > 0 else ""
            y_offset = win_rate_player1[x - 1] + 2  # Offset for annotation
            plt.text(
                x, y_offset, f"{plus_sign}{change:.2f}%", color="cyan", fontsize=8, ha="center"
            )

    # Annotate rate changes for Player 2 win rate
    for idx, (x, change) in enumerate(zip(annotation_x, [None] + rate_change_player2)):
        if change is not None and x != 0:
            plus_sign = "+" if change > 0 else ""
            y_offset = win_rate_player2[x - 1] - 2  # Offset for annotation
            plt.text(
                x, y_offset, f"{plus_sign}{change:.2f}%", color="orange", fontsize=8, ha="center"
            )

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
    for i, (x, change) in enumerate(zip(iq_x, roc_iq_values)):
        if change is not None and x != 0:
            plus_sign = "+" if change > 0 else ""
            # Prevent division by zero in y_offset calculation
            y_offset = iq_values[i] + (0.05 * iq_values[i]) if iq_values[i] != 0 else iq_values[i] + 1
            plt.text(
                x, y_offset, f"{plus_sign}{change:.2f}", color="yellow", fontsize=8, ha="center"
            )

    # Display winner statistics and progress percentage in the title
    plt.title(
        f"Win Rates, Average Rewards, Average Turns, and IQ Over Time\n"
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
plot_data(winners, rewards, turns, min_reward, max_reward)
