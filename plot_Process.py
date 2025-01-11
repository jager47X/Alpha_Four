import matplotlib.pyplot as plt
import re

# Path to the log file
log_file_path = "Test/6/log4.txt"

# Initialize data containers
agent1_wins = []
agent2_wins = []
draws = []
win_rate_agent1 = []
win_rate_agent2 = []
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
plt.figure(figsize=(10, 6))

# Plot Agent1's win rate
plt.plot(agent1_wins, label="Number of Agent1 Wins", marker="o")

# Plot Agent2's win rate
plt.plot(agent2_wins, label="Number of Agent2 Wins", marker="s")

# Add labels, title, and legend
plt.xlabel("Game Index")
plt.ylabel("Win Rate (%)")
plt.title("Comparison of Agent1 and Agent2 Wins Over Time")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
