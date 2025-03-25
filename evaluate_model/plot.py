import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# --------------------- User Inputs and Log File Setup ---------------------
version = input("model version>> ") or "1"
log_dir = os.path.join("data", "logs", "train_logs", version)
log_file_path = os.path.join(log_dir, "train.log")
os.makedirs(log_dir, exist_ok=True)
print("Log file path:", log_file_path)

total_episodes = int(input("total_episodes>> ") or 1000000)
print("Total Episodes:", total_episodes)
interval = int(input("interval>> ") or 1000)
print("Interval:", interval)
annotate_on = (input("annotate_on (true/false)>> ").strip().lower() or "false") == "true"
print("Annotate:", annotate_on)

# --------------------- Parsing Function ---------------------
def parse_log_file(log_file_path):
    """
    Reads the log file and extracts:
      - winner, reward, turn, epsilon, mcts_level,
      - mcts_rate, dqn_rate, hybrid_rate.
    Returns lists for each metric along with min and max reward.
    """
    winners, rewards, turns, epsilons = [], [], [], []
    mcts_levels, mcts_used_rate_list = [], []
    dqn_rates, hybrid_rates = [], []
    min_reward, max_reward = float('inf'), float('-inf')
    
    try:
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Example log:
                # Episode 123: Winner=1,Win Rate=55.00%, Turn=20, Reward=0.75, EPSILON=0.123456, MCTS LEVEL=2, MCTS Rate:45.67%, DQN Rate:12.34%, HYBRID Rate:41.00%
                match = re.search(
                    r"Episode\s+\d+:"                          # Episode number
                    r"\s+Winner=(-?\d+),Win Rate=[\d.]+%,"       # Winner (win rate ignored)
                    r"\s+Turn=(\d+),\s+Reward=([-.\d]+),"         # Turn and reward
                    r"\s+EPSILON=([\d.e-]+),\s+MCTS LEVEL=(\d+),"  # Epsilon and MCTS LEVEL
                    r"\s+MCTS Rate:([\d.]+)%,"                    # MCTS Rate
                    r"\s+DQN Rate:([\d.]+)%,"                     # DQN Rate
                    r"\s+HYBRID Rate:([\d.]+)%"                   # HYBRID Rate
                    , line)
                if match:
                    winners.append(int(match.group(1)))
                    turns.append(int(match.group(2)))
                    reward = float(match.group(3))
                    rewards.append(reward)
                    epsilons.append(float(match.group(4)))
                    mcts_levels.append(int(match.group(5)))
                    mcts_used_rate_list.append(float(match.group(6)))
                    dqn_rates.append(float(match.group(7)))
                    hybrid_rates.append(float(match.group(8)))
                    min_reward = min(min_reward, reward)
                    max_reward = max(max_reward, reward)
    except FileNotFoundError:
        print(f"Error: The log file at '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{log_file_path}': {e}")
    
    return (winners, rewards, turns, epsilons, mcts_levels,
            mcts_used_rate_list, dqn_rates, hybrid_rates, min_reward, max_reward)

# --------------------- Aggregation Function ---------------------
def aggregate_data(winners, rewards, turns, mcts_used_rate, dqn_rates, hybrid_rates, interval):
    """
    Aggregates raw data into intervals.
    Returns:
      interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
      avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full.
    """
    total_games = len(winners)
    avg_winrates, avg_rewards, avg_turns = [], [], []
    avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage = [], [], []
    interval_x = []
    idx = 0
    while idx < total_games:
        end_idx = min(idx + interval, total_games)
        interval_winners = winners[idx:end_idx]
        interval_rewards = rewards[idx:end_idx]
        interval_turns = turns[idx:end_idx]
        interval_mcts = mcts_used_rate[idx:end_idx]
        interval_dqn = dqn_rates[idx:end_idx]
        interval_hybrid = hybrid_rates[idx:end_idx]
        
        if interval_winners:
            p2_wins = sum(1 for w in interval_winners if w == 2)
            interval_winrate = (p2_wins / len(interval_winners)) * 100.0
        else:
            interval_winrate = 0.0
        
        non_zero_rewards = [r for r in interval_rewards if r != 0]
        avg_reward = np.mean(non_zero_rewards) if non_zero_rewards else 0.0
        non_zero_turns = [t for t in interval_turns if t > 0]
        avg_turn = np.mean(non_zero_turns) if non_zero_turns else 0.0
        
        avg_winrates.append(interval_winrate)
        avg_rewards.append(avg_reward)
        avg_turns.append(avg_turn / 2)  # as in original adjustment
        avg_mcts_usage.append(np.mean(interval_mcts) if interval_mcts else 0.0)
        avg_dqn_usage.append(np.mean(interval_dqn) if interval_dqn else 0.0)
        avg_hybrid_usage.append(np.mean(interval_hybrid) if interval_hybrid else 0.0)
        interval_x.append(end_idx)
        idx = end_idx
        
    interval_x_full = [0] + interval_x
    avg_winrates_full = [0.0] + avg_winrates
    avg_rewards_full = [0.0] + avg_rewards
    avg_turns_full = [0.0] + avg_turns
    avg_mcts_usage_full = [0.0] + avg_mcts_usage
    avg_dqn_usage_full = [0.0] + avg_dqn_usage
    avg_hybrid_usage_full = [0.0] + avg_hybrid_usage
    
    return (interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
            avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full)

# --------------------- Plotting Functions ---------------------
def plot_figure1(agg_data, total_games, total_episodes, total_p2_wins, draws, min_reward, max_reward, interval, annotate_on=False):
    try:
        print("Starting Figure 1 plotting...")
        (interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
         avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full) = agg_data

        if not interval_x_full or not avg_winrates_full:
            print("Error: Aggregated data is empty. Check the log file and aggregation settings.")
            return

        print("Aggregated data points count:", len(interval_x_full))
        max_points = 50  # further downsampling for performance
        if len(interval_x_full) > max_points:
            print(f"Downsampling data from {len(interval_x_full)} points to {max_points} points...")
            indices = np.linspace(0, len(interval_x_full) - 1, max_points, dtype=int)
            interval_x_sample = np.array(interval_x_full)[indices]
            avg_winrates_sample = np.array(avg_winrates_full)[indices]
            avg_rewards_sample = np.array(avg_rewards_full)[indices]
            avg_turns_sample = np.array(avg_turns_full)[indices]
            avg_mcts_usage_sample = np.array(avg_mcts_usage_full)[indices]
            avg_dqn_usage_sample = np.array(avg_dqn_usage_full)[indices]
            avg_hybrid_usage_sample = np.array(avg_hybrid_usage_full)[indices]
        else:
            print("No downsampling needed.")
            interval_x_sample = np.array(interval_x_full)
            avg_winrates_sample = np.array(avg_winrates_full)
            avg_rewards_sample = np.array(avg_rewards_full)
            avg_turns_sample = np.array(avg_turns_full)
            avg_mcts_usage_sample = np.array(avg_mcts_usage_full)
            avg_dqn_usage_sample = np.array(avg_dqn_usage_full)
            avg_hybrid_usage_sample = np.array(avg_hybrid_usage_full)

        print("Creating Plotly Figure for Figure 1...")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_winrates_sample,
                                  mode='lines+markers', name='Ave Win Rate (%)',
                                  line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_rewards_sample,
                                  mode='lines+markers', name='Ave Reward',
                                  line=dict(color='yellow')))
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_turns_sample,
                                  mode='lines+markers', name='Ave Turns',
                                  line=dict(color='cyan')))
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_mcts_usage_sample,
                                  mode='lines+markers', name='Ave MCTS Usage (%)',
                                  line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_dqn_usage_sample,
                                  mode='lines+markers', name='Ave DQN Usage (%)',
                                  line=dict(color='magenta')))
        fig1.add_trace(go.Scatter(x=interval_x_sample, y=avg_hybrid_usage_sample,
                                  mode='lines+markers', name='Ave HYBRID Usage (%)',
                                  line=dict(color='orange')))

        if annotate_on and len(interval_x_sample) <= 50:
            print("Computing rate-of-change annotations...")
            def add_annotations(x_vals, y_vals, color, unit=""):
                roc_vals = np.insert(np.diff(y_vals), 0, np.nan)
                for i in range(1, len(x_vals)):
                    if np.isnan(roc_vals[i]):
                        continue
                    plus_sign = "+" if roc_vals[i] > 0 else ""
                    fig1.add_annotation(
                        x=x_vals[i],
                        y=y_vals[i],
                        text=f"{plus_sign}{roc_vals[i]:.2f}{unit}",
                        font=dict(color=color, size=10),
                        showarrow=False,
                        xanchor="center",
                        yanchor="bottom"
                    )
            add_annotations(interval_x_sample, avg_winrates_sample, "green", "%")
            add_annotations(interval_x_sample, avg_rewards_sample, "yellow")
            add_annotations(interval_x_sample, avg_turns_sample, "cyan")
            add_annotations(interval_x_sample, avg_mcts_usage_sample, "red", "%")
            add_annotations(interval_x_sample, avg_dqn_usage_sample, "magenta", "%")
            add_annotations(interval_x_sample, avg_hybrid_usage_sample, "orange", "%")
        elif annotate_on:
            print("Too many points to annotate without affecting performance.")

        fig1.update_layout(
            title=(f"Figure 1: Interval-Based Metrics (each {interval} cases)<br>"
                   f"Games Processed: {total_games}/{total_episodes} "
                   f"({(interval_x_full[-1] / total_episodes)*100:.2f}% of target) | "
                   f"Ave Win Rate: {(total_p2_wins / total_games)*100:.2f}% | Draws: {draws} | "
                   f"Min Reward: {min_reward:.2f} | Max Reward: {max_reward:.2f}"),
            xaxis_title="Game Index (Interval End)",
            yaxis_title="Metric Value",
            template="plotly_dark"
        )
        print("Displaying Figure 1 in offline mode...")
        pyo.plot(fig1, auto_open=True)
        print("Figure 1 loaded successfully.")
    except Exception as e:
        print("An error occurred while plotting Figure 1:", str(e))


def plot_figure2(interval_x_full, avg_winrates_full, mcts_levels, total_games, total_episodes, total_p2_wins, draws, interval, epsilons):
    try:
        print("Starting Figure 2 plotting...")
        x_games = np.arange(1, total_games + 1)
        epsilon_percent = [e * 100 for e in epsilons]
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=x_games, y=epsilon_percent,
                                  mode='lines', name="Epsilon (%)", line=dict(color='blue')),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=interval_x_full, y=avg_winrates_full,
                                  mode='lines+markers', name="Ave Win Rate (%) [interval]", line=dict(color='green')),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=x_games, y=mcts_levels,
                                  mode='lines', name="MCTS Level", line=dict(color='white')),
                       secondary_y=True)
        fig2.update_layout(
            title=(f"Figure 2: Per-Game Metrics (each {interval} cases)<br>"
                   f"Games Processed: {total_games}/{total_episodes} "
                   f"({(interval_x_full[-1] / total_episodes)*100:.2f}% of target) | "
                   f"Ave Win Rate: {(total_p2_wins / total_games)*100:.2f}% | Draws: {draws}"),
            xaxis_title="Game Index",
            template="plotly_dark"
        )
        fig2.update_yaxes(title_text="Epsilon (%) / Ave Win Rate (%)", secondary_y=False)
        fig2.update_yaxes(title_text="MCTS Level", secondary_y=True)
        print("Displaying Figure 2 in offline mode...")
        pyo.plot(fig2, auto_open=True)
        print("Figure 2 loaded successfully.")
    except Exception as e:
        print("An error occurred while plotting Figure 2:", str(e))


def plot_figure3(avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage, avg_winrates):
    try:
        print("Starting Figure 3 plotting...")
        def add_trend_line(x, y, window=5):
            x = np.array(x)
            y = np.array(y)
            if len(x) < window:
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = poly(x_line)
                return x_line, y_line
            def moving_average(a, n):
                return np.convolve(a, np.ones(n)/n, mode='valid')
            x_smoothed = moving_average(x, window)
            y_smoothed = moving_average(y, window)
            coeffs = np.polyfit(x_smoothed, y_smoothed, 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = poly(x_line)
            return x_line, y_line

        fig3 = make_subplots(rows=1, cols=3,
                             subplot_titles=("MCTS vs Ave Win Rate", "DQN vs Ave Win Rate", "HYBRID vs Ave Win Rate"))
        # MCTS subplot
        fig3.add_trace(go.Scatter(x=avg_mcts_usage, y=avg_winrates,
                                  mode='markers', marker=dict(color='red'),
                                  name="MCTS Usage"), row=1, col=1)
        x_line, y_line = add_trend_line(avg_mcts_usage, avg_winrates)
        fig3.add_trace(go.Scatter(x=x_line, y=y_line,
                                  mode='lines', line=dict(color='red', dash='dash'),
                                  name="Trend (MCTS)"), row=1, col=1)
        # DQN subplot
        fig3.add_trace(go.Scatter(x=avg_dqn_usage, y=avg_winrates,
                                  mode='markers', marker=dict(color='magenta'),
                                  name="DQN Usage"), row=1, col=2)
        x_line, y_line = add_trend_line(avg_dqn_usage, avg_winrates)
        fig3.add_trace(go.Scatter(x=x_line, y=y_line,
                                  mode='lines', line=dict(color='magenta', dash='dash'),
                                  name="Trend (DQN)"), row=1, col=2)
        # HYBRID subplot
        fig3.add_trace(go.Scatter(x=avg_hybrid_usage, y=avg_winrates,
                                  mode='markers', marker=dict(color='orange'),
                                  name="HYBRID Usage"), row=1, col=3)
        x_line, y_line = add_trend_line(avg_hybrid_usage, avg_winrates)
        fig3.add_trace(go.Scatter(x=x_line, y=y_line,
                                  mode='lines', line=dict(color='orange', dash='dash'),
                                  name="Trend (HYBRID)"), row=1, col=3)
        for col in [1, 2, 3]:
            fig3.update_xaxes(title_text="Ave Strategy Usage (%)", row=1, col=col)
            fig3.update_yaxes(title_text="Ave Win Rate (%)", row=1, col=col)
        fig3.update_layout(title_text="Figure 3: Relationship Between Strategy Usage and Ave Win Rate",
                           template="plotly_dark")
        print("Displaying Figure 3 in offline mode...")
        pyo.plot(fig3, auto_open=True)
        print("Figure 3 loaded successfully.")
    except Exception as e:
        print("An error occurred while plotting Figure 3:", str(e))


def plot_figure4(avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage, avg_winrates, interval_x):
    try:
        print("Starting Figure 4 plotting...")
        # Prepare aggregated game indexes and corresponding data arrays
        game_ix = np.array(interval_x)  # aggregated game indexes (exclude initial 0)
        mcts = np.array(avg_mcts_usage)
        dqn = np.array(avg_dqn_usage)
        hybrid = np.array(avg_hybrid_usage)
        winrates = np.array(avg_winrates)
        
        # Compute overall min and max of game index.
        min_game, max_game = game_ix.min(), game_ix.max()
        
        # Precompute filtered data for several percentage ranges.
        filter_options = {}
        filter_options["All"] = {
            "x": mcts,
            "y": dqn,
            "z": hybrid,
            "text": [f"Game Index: {gi}" for gi in game_ix]
        }
        for i in range(10):
            lower = min_game + (max_game - min_game) * (i / 10)
            upper = min_game + (max_game - min_game) * ((i + 1) / 10)
            indices = np.where((game_ix >= lower) & (game_ix <= upper))[0]
            filter_options[f"{i*10}-{(i+1)*10}%"] = {
                "x": mcts[indices] if indices.size > 0 else [],
                "y": dqn[indices] if indices.size > 0 else [],
                "z": hybrid[indices] if indices.size > 0 else [],
                "text": [f"Game Index: {game_ix[j]}" for j in indices] if indices.size > 0 else []
            }
        
        # Create slider steps for filtering data by game index range.
        slider_steps = []
        for label, data in filter_options.items():
            step = {
                "args": [{"x": [data["x"]],
                          "y": [data["y"]],
                          "z": [data["z"]],
                          "text": [data["text"]]}],
                "label": label,
                "method": "restyle"
            }
            slider_steps.append(step)
        
        # Create initial trace using the "All" data.
        init_data = filter_options["All"]
        fig4 = go.Figure(data=[go.Scatter3d(
            x=init_data["x"],
            y=init_data["y"],
            z=init_data["z"],
            mode='markers',
            marker=dict(
                size=8,
                color=winrates,  # Marker color: Ave Win Rate (%)
                colorscale='Viridis',
                colorbar=dict(title="Ave Win Rate (%)")
            ),
            text=init_data["text"]
        )])
        fig4.update_layout(
            scene=dict(
                xaxis_title="Ave MCTS Usage (%)",
                yaxis_title="Ave DQN Usage (%)",
                zaxis_title="Ave HYBRID Usage (%)"
            ),
            title="Figure 4: 3D Scatter Plot (Game Index Filter)",
            template="plotly_dark",
            scene_aspectmode="cube",
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        # Add an interactive slider at the bottom center.
        fig4.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Game Index Filter: "},
                "pad": {"t": 50},
                "x": 0.5,
                "xanchor": "center",
                "y": 0,
                "yanchor": "top",
                "steps": slider_steps
            }]
        )
        print("Displaying Figure 4 in offline mode...")
        pyo.plot(fig4, auto_open=True)
        print("Figure 4 loaded successfully.")
    except Exception as e:
        print("An error occurred while plotting Figure 4:", str(e))


# --------------------- Main Execution ---------------------
(winners, rewards, turns, epsilons, mcts_levels, mcts_used_rate,
 dqn_rates, hybrid_rates, min_reward, max_reward) = parse_log_file(log_file_path)

if len(winners) == 0:
    print("No valid data parsed. Please check your log file format.")
else:
    total_games = len(winners)
    total_p2_wins = sum(1 for w in winners if w == 2)
    draws = winners.count(-1)
    
    # Aggregate data once.
    agg_data = aggregate_data(winners, rewards, turns, mcts_used_rate, dqn_rates, hybrid_rates, interval)
    (interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
     avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full) = agg_data
    
    print("Aggregated data points for Figure 1:", len(interval_x_full))
    
    # Plot Figure 1
    plot_figure1(agg_data, total_games, total_episodes, total_p2_wins, draws, min_reward, max_reward, interval, annotate_on)
    
    # Plot Figure 2
    plot_figure2(interval_x_full, avg_winrates_full, mcts_levels, total_games, total_episodes, total_p2_wins, draws, interval, epsilons)
    
    # Plot Figure 3 (using aggregated data without the initial 0)
    plot_figure3(np.array(avg_mcts_usage_full[1:]), np.array(avg_dqn_usage_full[1:]),
                 np.array(avg_hybrid_usage_full[1:]), np.array(avg_winrates_full[1:]))
    
    # Plot Figure 4 (using aggregated data without the initial 0)
    plot_figure4(np.array(avg_mcts_usage_full[1:]), np.array(avg_dqn_usage_full[1:]),
                 np.array(avg_hybrid_usage_full[1:]), np.array(avg_winrates_full[1:]),
                 np.array(interval_x_full[1:]))
