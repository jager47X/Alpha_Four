import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# Path to the log file
log_file_path = "log.txt"

# Initialize total rewards and outcomes lists
total_rewards = []
outcomes = []

# Read the log file and extract total rewards and outcomes
with open(log_file_path, "r") as log_file:
    for line in log_file:
        # Extract total rewards
        if "Total Reward:" in line:
            try:
                total_reward_value = float(line.split("Total Reward:")[1].split("|")[0].strip())
                total_rewards.append(total_reward_value)
            except ValueError:
                pass
        # Extract outcomes
        elif "Agent wins" in line:
            outcomes.append("win")
        elif "Agent lost" in line:
            outcomes.append("lose")
        elif "It's a draw" in line:
            outcomes.append("draw")

# Calculate win/loss stats
agent_wins = outcomes.count("win")
agent_losses = outcomes.count("lose")
agent_draws = outcomes.count("draw")
total_games = len(outcomes)
win_ratio = agent_wins / total_games if total_games > 0 else 0.0

# Default moving average window size
window = 100

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter_plot, = ax.plot([], [], 'bo', markersize=5, label="Total Rewards")
moving_avg_line, = ax.plot([], [], 'orange', linewidth=2, label="Moving Average")

# Display win/loss stats
stats_text = f"Agent Wins: {agent_wins} | Agent Losses: {agent_losses} | Agent Draws: {agent_draws} | Win Ratio: {win_ratio:.2f}"
ax.set_title(f"Training Progress: Total Rewards per Episode\n{stats_text}")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

# Plot update function
def update_plot(new_window):
    global total_rewards
    if len(total_rewards) == 0:
        return
    if len(total_rewards) < new_window:
        print("Warning: Window size exceeds total rewards length. Adjusting window size.")
        new_window = len(total_rewards)
    
    moving_avg = np.convolve(total_rewards, np.ones(new_window) / new_window, mode='valid')
    scatter_plot.set_data(range(len(total_rewards)), total_rewards)
    moving_avg_line.set_data(range(len(moving_avg)), moving_avg)
    
    # Dynamically adjust the Y-axis
    data_min = min(total_rewards)
    data_max = max(total_rewards)
    ax.set_ylim(data_min - 1, data_max + 1)

    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"Training Progress: Total Rewards per Episode\n{stats_text}")
    plt.draw()

# Button callback functions
def increase_window(event):
    global window
    window += 10
    update_plot(window)

def decrease_window(event):
    global window
    if window > 10:
        window -= 10
        update_plot(window)

# Add buttons
ax_increase = plt.axes([0.7, 0.01, 0.1, 0.05])  # [x, y, width, height]
btn_increase = Button(ax_increase, 'Increase')

ax_decrease = plt.axes([0.81, 0.01, 0.1, 0.05])
btn_decrease = Button(ax_decrease, 'Decrease')

btn_increase.on_clicked(increase_window)
btn_decrease.on_clicked(decrease_window)

# Initial plot
update_plot(window)
plt.show()
