import matplotlib.pyplot as plt
import re

# Path to the log file
log_file_path = "Test/6/log1.txt"

# Initialize data containers
agent1_wins = []
agent2_wins = []
draws = []

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
                agent1_wins.append(agent1_win_count)
                agent2_wins.append(agent2_win_count)
                draws.append(draw_count)

except FileNotFoundError:
    print(f"Error: The log file at '{log_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Plot the data
plt.figure(figsize=(10, 6))

# Plot Agent1's win count
plt.plot(agent1_wins, label="Number of Agent1 Wins", marker=".", color="blue")

# Plot Agent2's win count
plt.plot(agent2_wins, label="Number of Agent2 Wins", marker="s", color="orange")

# Plot Draw count
plt.plot(draws, label="Number of Draws", marker="^", linestyle="--", color="green")


# Add labels, title, and legend
plt.xlabel("Game Index")
plt.ylabel("Count")
plt.title("Comparison of Agent1 Wins, Agent2 Wins, and Draws Over Time")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
