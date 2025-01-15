import matplotlib.pyplot as plt
import re
import numpy as np
import time

# Path to the log file
log_file_path = input("FILE PATH>> ")

# Function to parse the log file
def parse_log_file(log_file_path):
    rewards = []
    winners = []

    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Extract Winner and Reward data from the log
                match = re.search(r"Winner=(-?\d+), Reward=([-.\d]+)", line)  # Matches "Winner=X, Reward=Y"
                if match:
                    winner = int(match.group(1))  # Winner (1, 2, or -1 for draws)
                    reward = float(match.group(2))  # Reward

                    # Append the extracted values
                    winners.append(winner)
                    rewards.append(reward)
                else:
                    # Skip lines that don't match the pattern
                    continue

    except FileNotFoundError:
        print(f"Error: The log file at '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return winners, rewards

# Function to plot the data
def plot_data(winners, rewards):
    if not rewards or not winners:
        print("No data to plot.")
        return

    # Calculate cumulative statistics
    total_games = len(winners)
    winner_counts = {
        1: winners.count(1),   # Wins for Player 1
        2: winners.count(2),   # Wins for Player 2
        -1: winners.count(-1)  # Draws
    }

    # Compute cumulative win rates
    cumulative_games = np.arange(1, total_games + 1)  # Game indices
    cumulative_wins_player1 = np.cumsum(np.array(winners) == 1)
    cumulative_wins_player2 = np.cumsum(np.array(winners) == 2)

    win_rate_player1 = cumulative_wins_player1 / cumulative_games * 100
    win_rate_player2 = cumulative_wins_player2 / cumulative_games * 100

    # Compute average rewards for every 100 games
    interval = 100
    avg_rewards = [
        np.mean(rewards[start:min(start + interval, len(rewards))])
        for start in range(0, len(rewards), interval)
    ]
    avg_rewards_x = list(range(interval, interval * len(avg_rewards) + 1, interval))

    # Compute win rate changes for annotations
    rate_change_player1 = [
        win_rate_player1[min(start + interval - 1, total_games - 1)] - win_rate_player1[start]
        for start in range(0, total_games, interval)
    ]
    rate_change_player2 = [
        win_rate_player2[min(start + interval - 1, total_games - 1)] - win_rate_player2[start]
        for start in range(0, total_games, interval)
    ]
    annotation_x = list(range(interval, total_games + 1, interval))

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot win rates
    plt.plot(cumulative_games, win_rate_player1, label="Player 1 Win Rate (%)", marker="o", color="blue")
    plt.plot(cumulative_games, win_rate_player2, label="Player 2 Win Rate (%)", marker="s", color="orange")

    # Annotate rate changes for Player 1
    for x, change in zip(annotation_x, rate_change_player1):
        plus_sign = "+" if change > 0 else ""
        y_offset = win_rate_player1[x - 1] + 4  # Offset for annotation
        plt.text(
            x, y_offset, f"{plus_sign}{change:.2f}%", color="blue", fontsize=8, ha="center"
        )

    # Annotate rate changes for Player 2
    for x, change in zip(annotation_x, rate_change_player2):
        plus_sign = "+" if change > 0 else ""
        y_offset = win_rate_player2[x - 1] - 4  # Offset for annotation
        plt.text(
            x, y_offset, f"{plus_sign}{change:.2f}%", color="orange", fontsize=8, ha="center"
        )

    # Plot average rewards
    plt.plot(avg_rewards_x, avg_rewards, label="Average Reward per 100 Games", marker="x", color="green")

    # Display winner statistics in the title
    plt.title(
        f"Win Rates and Average Rewards Over Time\n"
        f"Total Games: {total_games}, "
        f"Player 1 Wins: {winner_counts.get(1, 0)}, "
        f"Player 2 Wins: {winner_counts.get(2, 0)}, "
        f"Draws: {winner_counts.get(-1, 0)}"
    )
    plt.xlabel("Game Index")
    plt.ylabel("Percentage / Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Periodically update every 10 seconds
try:
    while True:
        print("Updating plot...")
        winners, rewards = parse_log_file(log_file_path)
        plot_data(winners, rewards)
        time.sleep(10)  # Wait 10 seconds before refreshing
except KeyboardInterrupt:
    print("Exiting...")
