import matplotlib.pyplot as plt
import re
import numpy as np

# Path to the log file
log_file_path = input("FILE PATH>> ")

# Initialize data containers
agent1_wins = []
agent2_wins = []
draws = []
total_games = []

# Read and parse the log file
try:
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            # ------------------------------------------------------
            # We attempt up to 3 regex patterns in succession:
            #   1) "Agent1 wins=XX ... Agent2 wins=YY ... draws=ZZ"
            #   2) "Trainer wins=XX ... Agent wins=YY ... draws=ZZ"
            #   3) "R=## ... Wins=XX ... OppWins=YY ... Draws=ZZ"
            #
            # We'll parse:
            #   a1 = agent1 or trainer wins
            #   a2 = agent2 or agent wins
            #   d  = draws
            # ------------------------------------------------------
            match = re.search(r"Agent1\s+wins=(\d+).*Agent2\s+wins=(\d+).*draws=(\d+)", line)
            if match:
                a1 = int(match.group(1))  # agent1 wins
                a2 = int(match.group(2))  # agent2 wins
                d  = int(match.group(3))  # draws

            else:
                match = re.search(r"Trainer\s+wins=(\d+).*Agent\s+wins=(\d+).*draws=(\d+)", line)
                if match:
                    a1 = int(match.group(1))  # trainer wins
                    a2 = int(match.group(2))  # agent wins
                    d  = int(match.group(3))  # draws
                else:
                    match = re.search(r"R=\d+.*Wins=(\d+).*OppWins=(\d+).*Draws=(\d+)", line)
                    if match:
                        a1 = int(match.group(1))  # "Wins" for agent1
                        a2 = int(match.group(2))  # "OppWins" for agent2
                        d  = int(match.group(3))  # "Draws"
                    else:
                        # No match => skip this line
                        continue

            # If we reach here, we successfully extracted a1, a2, d
            total = a1 + a2 + d
            agent1_wins.append(a1)
            agent2_wins.append(a2)
            draws.append(d)
            total_games.append(total)

except FileNotFoundError:
    print(f"Error: The log file at '{log_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# -------------------------------------------------------------------
# If no data was parsed, we can't plot
# -------------------------------------------------------------------
if not total_games:
    print("No matching data was found in the log. Exiting.")
    exit()

# Calculate cumulative values
cumulative_games = np.cumsum(total_games)
cumulative_agent1_wins = np.cumsum(agent1_wins)
cumulative_agent2_wins = np.cumsum(agent2_wins)

# Calculate win rates (as percentages)
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

    # Difference in win rate over this interval
    rate_change1 = agent1_win_rate[end - 1] - agent1_win_rate[start]
    rate_change2 = agent2_win_rate[end - 1] - agent2_win_rate[start]

    agent1_rate_of_change.append(rate_change1)
    agent2_rate_of_change.append(rate_change2)
    interval_midpoints.append(midpoint)

# Calculate overall average win/draw rates
sum_total = sum(total_games)
agent1_average = round(sum(agent1_wins) / sum_total * 100, 2)
agent2_average = round(sum(agent2_wins) / sum_total * 100, 2)
draw_average   = round(sum(draws) / sum_total * 100, 2)

# --- Plotting ---
plt.figure(figsize=(14, 8))

# Plot Agent1's win rate
plt.plot(
    agent1_win_rate,
    label=f"Agent1 Wins, Avg-Win ({agent1_average}%)",
    marker=".",
    color="blue",
)

# Plot Agent2's win rate
plt.plot(
    agent2_win_rate,
    label=f"Agent2 Wins, Avg-Win ({agent2_average}%)",
    marker="s",
    color="orange",
)

# Annotate rate of change for Agent1
for midpoint, rate_change in zip(interval_midpoints, agent1_rate_of_change):
    plus_sign = "+" if rate_change > 0 else ""
    y_offset = agent1_win_rate[midpoint] - 3
    plt.text(
        midpoint,
        y_offset,
        f"{plus_sign}{rate_change:.2f}%",
        color="blue",
        fontsize=9,
        ha="center",
    )

# Annotate rate of change for Agent2
for midpoint, rate_change in zip(interval_midpoints, agent2_rate_of_change):
    plus_sign = "+" if rate_change > 0 else ""
    y_offset = agent2_win_rate[midpoint] - 3
    plt.text(
        midpoint,
        y_offset,
        f"{plus_sign}{rate_change:.2f}%",
        color="orange",
        fontsize=9,
        ha="center",
    )

plt.title("Win Rates Over Time with Annotated Rate of Change")
plt.xlabel("Game Index")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
