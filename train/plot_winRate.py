import matplotlib.pyplot as plt
import re
import numpy as np

# Path to the log file
log_file_path = input("FILE PATH>> ")

# Function to parse the log file
def parse_log_file(log_file_path):
    rewards = []
    winners = []
    min_reward = float('inf')  # Initialize to a large value
    max_reward = float('-inf')  # Initialize to a small value

    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Extract Winner and Reward data from the log
                match = re.search(r"Winner=(-?\d+), Reward=([-.\d]+)", line)  # Matches "Winner=X, Reward=Y"
                if match:
                    winner = int(match.group(1))  # Winner (1, 2, or -1 for draws)
                    reward = float(match.group(2))  # Reward

                    # Update min and max rewards
                    if reward > max_reward:
                        max_reward = reward
                    if reward < min_reward:
                        min_reward = reward

                    # Append the extracted values
                    winners.append(winner)
                    rewards.append(reward)

    except FileNotFoundError:
        print(f"Error: The log file at '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return winners, rewards, min_reward, max_reward

# Function to plot the data
def plot_data(winners, rewards, min_reward, max_reward, total_episodes=100000):
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

    # Compute average rewards for every 1000 games
    interval_avg = 1000
    avg_rewards = [
        np.mean(rewards[start:min(start + interval_avg, len(rewards))])
        for start in range(0, len(rewards), interval_avg)
    ]
    avg_rewards_x = list(range(interval_avg, interval_avg * len(avg_rewards) + 1, interval_avg))

    # Compute win rate changes for annotations
    interval_annotate = 1000
    rate_change_player1 = [
        win_rate_player1[min(start + interval_annotate - 1, total_games - 1)] - win_rate_player1[start]
        for start in range(0, total_games, interval_annotate)
    ]
    rate_change_player2 = [
        win_rate_player2[min(start + interval_annotate - 1, total_games - 1)] - win_rate_player2[start]
        for start in range(0, total_games, interval_annotate)
    ]
    annotation_x = list(range(interval_annotate, total_games + 1, interval_annotate))

    # Calculate progress percentage
    progress_percentage = (total_games / total_episodes) * 100

    # Set black background style
    plt.style.use('dark_background')

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot win rates
    plt.plot(cumulative_games, win_rate_player1, label="Player 1 Win Rate (%)", marker="o", color="cyan")
    plt.plot(cumulative_games, win_rate_player2, label="Player 2 Win Rate (%)", marker="s", color="orange")

    # Annotate rate changes for Player 1
    for x, change in zip(annotation_x, rate_change_player1):
        plus_sign = "+" if change > 0 else ""
        y_offset = win_rate_player1[x - 1] + 4  # Offset for annotation
        plt.text(
            x, y_offset, f"{plus_sign}{change:.2f}%", color="cyan", fontsize=8, ha="center"
        )

    # Annotate rate changes for Player 2
    for x, change in zip(annotation_x, rate_change_player2):
        plus_sign = "+" if change > 0 else ""
        y_offset = win_rate_player2[x - 1] - 4  # Offset for annotation
        plt.text(
            x, y_offset, f"{plus_sign}{change:.2f}%", color="orange", fontsize=8, ha="center"
        )

    # Plot average rewards
    plt.plot(avg_rewards_x, avg_rewards, label="Average Reward per 1000 Games", marker="x", color="lime")

    # Display winner statistics and progress percentage in the title
    plt.title(
        f"Win Rates and Average Rewards Over Time\n"
        f"Total Games: {total_games}/{total_episodes} ({progress_percentage:.2f}% Completed), "
        f"Player 1 Wins: {winner_counts.get(1, 0)}, "
        f"Player 2 Wins: {winner_counts.get(2, 0)}, "
        f"Agent MIN Reward: {min_reward:.2f}, "
        f"Agent MAX Reward: {max_reward:.2f}, "
        f"Draws: {winner_counts.get(-1, 0)}"
    )
    plt.xlabel("Game Index")
    plt.ylabel("Percentage / Reward")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Parse the log file and plot the data
winners, rewards, min_reward, max_reward = parse_log_file(log_file_path)
plot_data(winners, rewards, min_reward, max_reward)
