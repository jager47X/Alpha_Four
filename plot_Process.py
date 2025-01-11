import matplotlib.pyplot as plt
import re

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

# Plot the data
plt.figure(figsize=(12, 7))

# Plot Agent1's win count
plt.plot(agent1_wins, label="Number of Agent1 Wins", marker=".", color="blue")

# Plot Agent2's win count
plt.plot(agent2_wins, label="Number of Agent2 Wins", marker="s", color="orange")

# Plot Draw count
plt.plot(draws, label="Number of Draws", marker="^", linestyle="--", color="green")

# Set interval for non-cumulative calculation
interval = 100

# Loop through intervals to compute win rates for each segment
for start in range(0, len(total_games), interval):
    end = min(start + interval, len(total_games))  # Define the segment end
    segment_total_games = total_games[end - 1] - (total_games[start - 1] if start > 0 else 0)
    segment_agent1_wins = agent1_wins[end - 1] - (agent1_wins[start - 1] if start > 0 else 0)
    segment_agent2_wins = agent2_wins[end - 1] - (agent2_wins[start - 1] if start > 0 else 0)

    if segment_total_games > 0:  # Avoid division by zero
        win_rate_agent1 = (segment_agent1_wins / segment_total_games) * 100
        win_rate_agent2 = (segment_agent2_wins / segment_total_games) * 100

        # Annotate the segment win rates
        midpoint = (start + end) // 2  # Annotate at the middle of the segment
        plt.text(midpoint, agent1_wins[end - 1] + 10, f"{win_rate_agent1:.1f}%", color="blue", fontsize=9, ha="center")
        plt.text(midpoint, agent2_wins[end - 1] + 10, f"{win_rate_agent2:.1f}%", color="orange", fontsize=9, ha="center")


# Add labels, title, and legend
plt.xlabel("Game Index")
plt.ylabel(f"Count, interval set to {interval}")
plt.title("Comparison of Agent1 Wins, Agent2 Wins, and Draws Over Time with Win Rate Annotations")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
