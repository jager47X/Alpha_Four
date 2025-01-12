import matplotlib.pyplot as plt
import re
import numpy as np

# Path to the log file
log_file_path = input("FILE PATH>>")

# Initialize data containers
agent1_wins = []
agent2_wins = []
draws = []
total_games = []

# Read and parse the log file
try:
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            # Extract wins, rewards, and draws using regular expressions
            match = re.search(r"Agent1 wins=(\d+).*Agent2 wins=(\d+).*draws=(\d+)", line)
            if not match:
                match = re.search(r"Trainer wins=(\d+).*Agent wins=(\d+).*draws=(\d+)", line)
            if match:
                agent1_win_count = int(match.group(1))
                agent2_win_count = int(match.group(2))
                draw_count = int(match.group(3))

                # Update the data
                total = agent1_win_count + agent2_win_count + draw_count
                agent1_wins.append(agent1_win_count)
                agent2_wins.append(agent2_win_count)
                draws.append(draw_count)
                total_games.append(total)

except FileNotFoundError:
    print(f"Error: The log file at '{log_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Calculate cumulative values
cumulative_games = np.cumsum(total_games)
cumulative_agent1_wins = np.cumsum(agent1_wins)
cumulative_agent2_wins = np.cumsum(agent2_wins)

# Calculate win rates
agent1_win_rate = cumulative_agent1_wins / cumulative_games * 100
agent2_win_rate = cumulative_agent2_wins / cumulative_games * 100

# Compute rate of change for each interval
interval = 100
agent1_rate_of_change = []
agent2_rate_of_change = []
interval_midpoints = []

for start in range(0, len(agent1_win_rate), interval):
    end = min(start + interval, len(agent1_win_rate))
    midpoint = (start + end) // 2  # Midpoint for annotation

    rate_change1 = agent1_win_rate[end - 1] - agent1_win_rate[start]
    rate_change2 = agent2_win_rate[end - 1] - agent2_win_rate[start]

    agent1_rate_of_change.append(rate_change1)
    agent2_rate_of_change.append(rate_change2)
    interval_midpoints.append(midpoint)

# Calculate the cumulative averages
agent1_average = round(sum(agent1_wins) / sum(total_games) * 100, 2)
agent2_average = round(sum(agent2_wins) / sum(total_games) * 100, 2)
draw_average = round(sum(draws) / sum(total_games) * 100, 2)

# Figure: Win Rates with Annotated Rate of Change
plt.figure(figsize=(14, 8))

# Plot Agent1's win rate
plt.plot(agent1_win_rate, label=f"Agent1 Wins, Avg-Win ({agent1_average}%)", marker=".", color="blue")

# Plot Agent2's win rate
plt.plot(agent2_win_rate, label=f"Agent2 Wins, Avg-Win ({agent2_average}%)", marker="s", color="orange")

# Annotate rate of change for Agent1
for midpoint, rate_change in zip(interval_midpoints, agent1_rate_of_change):
    plus_sign = "+" if rate_change > 0 else ""
    plt.text(
        midpoint,
        agent1_win_rate[midpoint]-3,  # Position at the current win rate
        f"{plus_sign}{rate_change:.4f}%",
        color="blue",
        fontsize=9,
        ha="center",
    )

# Annotate rate of change for Agent2
for midpoint, rate_change in zip(interval_midpoints, agent2_rate_of_change):
    plus_sign = "+" if rate_change > 0 else ""
    plt.text(
        midpoint,
        agent2_win_rate[midpoint] -3,  # Position at the current win rate
        f"{plus_sign}{rate_change:.4f}%",
        color="orange",
        fontsize=9,
        ha="center",
    )

# Add labels, title, and legend
plt.title("Win Rates Over Time with Annotated Rate of Change")
plt.xlabel("Game Index")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
