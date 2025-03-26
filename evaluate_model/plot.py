import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# --------------------- User Inputs and Log File Setup ---------------------
version = input("model version>> ") or "1"
log_dir = os.path.join("data", "logs", "train_logs", version)
log_file_path = os.path.join(log_dir, "train.log")
os.makedirs(log_dir, exist_ok=True)
print("Log file path:", log_file_path)

# Create the plots directory for the given version:
plots_dir = os.path.join("data", "models", "plots", version)
os.makedirs(plots_dir, exist_ok=True)
print("Plots will be saved in:", plots_dir)

# Fixed intervals:
FIG1_INTERVAL = 100    # For Figure 1
FIG2_INTERVAL = 1000   # For Figure 2 (and used for Figures 3 & 4 win rate aggregation)

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
                # Expected log line format:
                # Episode 123: Winner=1,Win Rate=55.00%, Turn=20, Reward=0.75, EPSILON=0.123456, MCTS LEVEL=2, MCTS Rate:45.67%, DQN Rate:12.34%, HYBRID Rate:41.00%
                match = re.search(
                    r"Episode\s+\d+:"                          
                    r"\s+Winner=(-?\d+),Win Rate=[\d.]+%,"       
                    r"\s+Turn=(\d+),\s+Reward=([-.\d]+),"         
                    r"\s+EPSILON=([\d.e-]+),\s+MCTS LEVEL=(\d+),"  
                    r"\s+MCTS Rate:([\d.]+)%,"                    
                    r"\s+DQN Rate:([\d.]+)%,"                     
                    r"\s+HYBRID Rate:([\d.]+)%"                    
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
        avg_turns.append(avg_turn / 2)
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
def plot_figure1(agg_data, total_games, total_p2_wins, draws, min_reward, max_reward, interval):
    try:
        print("Starting Figure 1 plotting...")
        (interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
         avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full) = agg_data
        print("Aggregated data points count:", len(interval_x_full))
        max_points = 50
        if len(interval_x_full) > max_points:
            indices = np.linspace(0, len(interval_x_full) - 1, max_points, dtype=int)
            interval_x_sample = np.array(interval_x_full)[indices]
            avg_winrates_sample = np.array(avg_winrates_full)[indices]
            avg_rewards_sample = np.array(avg_rewards_full)[indices]
            avg_turns_sample = np.array(avg_turns_full)[indices]
            avg_mcts_usage_sample = np.array(avg_mcts_usage_full)[indices]
            avg_dqn_usage_sample = np.array(avg_dqn_usage_full)[indices]
            avg_hybrid_usage_sample = np.array(avg_hybrid_usage_full)[indices]
        else:
            interval_x_sample = np.array(interval_x_full)
            avg_winrates_sample = np.array(avg_winrates_full)
            avg_rewards_sample = np.array(avg_rewards_full)
            avg_turns_sample = np.array(avg_turns_full)
            avg_mcts_usage_sample = np.array(avg_mcts_usage_full)
            avg_dqn_usage_sample = np.array(avg_dqn_usage_full)
            avg_hybrid_usage_sample = np.array(avg_hybrid_usage_full)
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
        fig1.update_layout(
            title=f"Figure 1: Interval-Based Metrics (Interval = {interval}) | Games: {total_games} | WinRate: {total_p2_wins/total_games*100:.2f}% | Draws: {draws} | Reward Range: [{min_reward:.2f}, {max_reward:.2f}]",
            xaxis_title="Game Index (Interval End)",
            yaxis_title="Metric Value",
            template="plotly_dark"
        )
        print("Displaying Figure 1 in offline mode...")
        pyo.plot(fig1, filename=os.path.join(plots_dir, "figure1.html"), auto_open=True)
    except Exception as e:
        print("An error occurred while plotting Figure 1:", str(e))


def plot_figure2(interval_x_full, avg_winrates_full, mcts_levels, epsilons):
    try:
        print("Starting Figure 2 plotting...")
        total_games = len(mcts_levels)
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
            title="Figure 2: Per-Game Metrics (Interval = 1000)",
            xaxis_title="Game Index",
            template="plotly_dark"
        )
        fig2.update_yaxes(title_text="Epsilon (%) / Ave Win Rate (%)", secondary_y=False)
        fig2.update_yaxes(title_text="MCTS Level", secondary_y=True)
        print("Displaying Figure 2 in offline mode...")
        pyo.plot(fig2, filename=os.path.join(plots_dir, "figure2.html"), auto_open=True)
    except Exception as e:
        print("An error occurred while plotting Figure 2:", str(e))

def plot_figure3(avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage, avg_winrates, game_index):
    try:
        print("Starting Figure 3 plotting with game index filter slider (using interval = 1000 data, game index scaled by 1/1000)...")
        def compute_trend_line(x, y):
            if len(x) < 2:
                return x, y
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = poly(x_line)
            return x_line, y_line

        # Scale game index by 1/1000
        game_ix = np.array(game_index) / 1000.0
        mcts = np.array(avg_mcts_usage)
        dqn = np.array(avg_dqn_usage)
        hybrid = np.array(avg_hybrid_usage)
        winrates = np.array(avg_winrates)
        
        min_game, max_game = game_ix.min(), game_ix.max()
        
        # Use percentage labels for the slider
        filter_options = {}
        filter_options["0-100%"] = {
            "mcts_x": mcts,
            "mcts_y": winrates,
            "dqn_x": dqn,
            "dqn_y": winrates,
            "hybrid_x": hybrid,
            "hybrid_y": winrates,
            "mcts_trend": compute_trend_line(mcts, winrates),
            "dqn_trend": compute_trend_line(dqn, winrates),
            "hybrid_trend": compute_trend_line(hybrid, winrates)
        }
        for i in range(10):
            lower = min_game + (max_game - min_game) * (i / 10)
            upper = min_game + (max_game - min_game) * ((i + 1) / 10)
            indices = np.where((game_ix >= lower) & (game_ix <= upper))[0]
            if indices.size == 0:
                filter_options[f"{i*10}-{(i+1)*10}%"] = {
                    "mcts_x": [],
                    "mcts_y": [],
                    "dqn_x": [],
                    "dqn_y": [],
                    "hybrid_x": [],
                    "hybrid_y": [],
                    "mcts_trend": ([], []),
                    "dqn_trend": ([], []),
                    "hybrid_trend": ([], [])
                }
            else:
                f_mcts = mcts[indices]
                f_dqn = dqn[indices]
                f_hybrid = hybrid[indices]
                f_winrates = winrates[indices]
                filter_options[f"{i*10}-{(i+1)*10}%"] = {
                    "mcts_x": f_mcts,
                    "mcts_y": f_winrates,
                    "dqn_x": f_dqn,
                    "dqn_y": f_winrates,
                    "hybrid_x": f_hybrid,
                    "hybrid_y": f_winrates,
                    "mcts_trend": compute_trend_line(f_mcts, f_winrates),
                    "dqn_trend": compute_trend_line(f_dqn, f_winrates),
                    "hybrid_trend": compute_trend_line(f_hybrid, f_winrates)
                }
        
        fig3 = make_subplots(rows=1, cols=3,
                             subplot_titles=("MCTS vs Ave Win Rate", "DQN vs Ave Win Rate", "HYBRID vs Ave Win Rate"))
        init = filter_options["0-100%"]
        fig3.add_trace(go.Scatter(x=init["mcts_x"], y=init["mcts_y"],
                                  mode='markers', marker=dict(color='red'),
                                  name="MCTS Usage"), row=1, col=1)
        mcts_trend_x, mcts_trend_y = init["mcts_trend"]
        fig3.add_trace(go.Scatter(x=mcts_trend_x, y=mcts_trend_y,
                                  mode='lines', line=dict(color='red', dash='dash'),
                                  name="Trend (MCTS)"), row=1, col=1)
        
        fig3.add_trace(go.Scatter(x=init["dqn_x"], y=init["dqn_y"],
                                  mode='markers', marker=dict(color='magenta'),
                                  name="DQN Usage"), row=1, col=2)
        dqn_trend_x, dqn_trend_y = init["dqn_trend"]
        fig3.add_trace(go.Scatter(x=dqn_trend_x, y=dqn_trend_y,
                                  mode='lines', line=dict(color='magenta', dash='dash'),
                                  name="Trend (DQN)"), row=1, col=2)
        
        fig3.add_trace(go.Scatter(x=init["hybrid_x"], y=init["hybrid_y"],
                                  mode='markers', marker=dict(color='orange'),
                                  name="HYBRID Usage"), row=1, col=3)
        hybrid_trend_x, hybrid_trend_y = init["hybrid_trend"]
        fig3.add_trace(go.Scatter(x=hybrid_trend_x, y=hybrid_trend_y,
                                  mode='lines', line=dict(color='orange', dash='dash'),
                                  name="Trend (HYBRID)"), row=1, col=3)
        
        for col in [1, 2, 3]:
            fig3.update_xaxes(title_text="Ave Strategy Usage (%)", row=1, col=col)
            fig3.update_yaxes(title_text="Ave Win Rate (%)", row=1, col=col)
        fig3.update_layout(title_text="Figure 3: Relationship Between Strategy Usage and Ave Win Rate (Interval = 1000)",
                           template="plotly_dark")
        
        slider_steps = []
        for label, data in filter_options.items():
            step = {
                "args": [{
                    "x": [data["mcts_x"], data["mcts_trend"][0],
                          data["dqn_x"], data["dqn_trend"][0],
                          data["hybrid_x"], data["hybrid_trend"][0]],
                    "y": [data["mcts_y"], data["mcts_trend"][1],
                          data["dqn_y"], data["dqn_trend"][1],
                          data["hybrid_y"], data["hybrid_trend"][1]]
                }],
                "label": label,
                "method": "restyle"
            }
            slider_steps.append(step)
        
        fig3.update_layout(
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
        print("Displaying Figure 3 in offline mode...")
        pyo.plot(fig3, filename=os.path.join(plots_dir, "figure3.html"), auto_open=True)
    except Exception as e:
        print("An error occurred while plotting Figure 3:", str(e))


def plot_figure4(agg_data, total_games, interval):
    try:
        print("Starting Figure 4 (3D scatter with slider filter) plotting...")
        (interval_x_full, avg_winrates_full, avg_rewards_full, avg_turns_full,
         avg_mcts_usage_full, avg_dqn_usage_full, avg_hybrid_usage_full) = agg_data
        game_ix = np.array(interval_x_full[1:]) / 1000.0  # scale by 1/1000
        mcts = np.array(avg_mcts_usage_full[1:])
        dqn = np.array(avg_dqn_usage_full[1:])
        hybrid = np.array(avg_hybrid_usage_full[1:])
        winrates = np.array(avg_winrates_full[1:])
        min_game, max_game = game_ix.min(), game_ix.max()
        filter_options = {}
        filter_options["0-100%"] = {
            "x": mcts,
            "y": dqn,
            "z": hybrid,
            "text": [f"Game Index: {gi:.2f}" for gi in game_ix]
        }
        for i in range(10):
            lower = min_game + (max_game - min_game) * (i / 10)
            upper = min_game + (max_game - min_game) * ((i + 1) / 10)
            indices = np.where((game_ix >= lower) & (game_ix <= upper))[0]
            filter_options[f"{i*10}-{(i+1)*10}%"] = {
                "x": mcts[indices] if indices.size > 0 else [],
                "y": dqn[indices] if indices.size > 0 else [],
                "z": hybrid[indices] if indices.size > 0 else [],
                "text": [f"Game Index: {game_ix[j]:.2f}" for j in indices] if indices.size > 0 else []
            }
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
        init_data = filter_options["0-100%"]
        fig4 = go.Figure(data=[go.Scatter3d(
            x=init_data["x"],
            y=init_data["y"],
            z=init_data["z"],
            mode='markers',
            marker=dict(
                size=8,
                color=winrates,
                colorscale='Viridis',
                colorbar=dict(title="Ave Win Rate (%)")
            ),
            text=init_data["text"]
        )])
        fig4.update_layout(
            title="Figure 4: 3D Scatter Cube with Game Index Filter (Interval = 1000, Game Index scaled by 1/1000)",
            scene=dict(
                xaxis_title="Ave MCTS Usage (%)",
                yaxis_title="Ave DQN Usage (%)",
                zaxis_title="Ave HYBRID Usage (%)"
            ),
            template="plotly_dark",
            scene_aspectmode="cube",
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
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
        pyo.plot(fig4, filename=os.path.join(plots_dir, "figure4.html"), auto_open=True)
    except Exception as e:
        print("An error occurred while plotting Figure 4:", str(e))


def plot_figure5(avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage, winners):
    try:
        print("Starting Figure 5 plotting (prediction outcome with slider)...")
        import numpy as np
        import plotly.graph_objects as go
        import plotly.offline as pyo
        from sklearn.neighbors import KNeighborsRegressor

        # Normalize inputs (assumed percentages)
        X_mcts = np.array(avg_mcts_usage) / 100.0
        X_dqn = np.array(avg_dqn_usage) / 100.0
        X_hybrid = np.array(avg_hybrid_usage) / 100.0
        y_winrates = np.where(np.array(winners) == 2, 1.0, 0.0)

        # Enforce MCTS + DQN + HYBRID = 1.0 exactly
        total = X_mcts + X_dqn + X_hybrid
        X_mcts /= total
        X_dqn /= total
        X_hybrid /= total

        # Create a normalized time array (scaled by 1/5)
        n = len(X_mcts)
        time_array = np.linspace(0, 1, n) / 5.0
        features = np.column_stack([X_mcts, X_dqn, X_hybrid, time_array])

        # Train KNN model
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(features, y_winrates)
        print("KNN model fitted on raw row data.")

        # Create prediction grid: Only combinations where x + y + z = 1.0
        steps = 100
        grid_vals = np.linspace(0, 1, steps + 1)
        X_list, Y_list, Z_list = [], [], []
        for x in grid_vals:
            for y in grid_vals:
                z = 1.0 - x - y
                if 0 <= z <= 1:
                    X_list.append(x)
                    Y_list.append(y)
                    Z_list.append(z)
        X_grid = np.array(X_list)
        Y_grid = np.array(Y_list)
        Z_grid = np.array(Z_list)
        grid_points = np.column_stack([X_grid, Y_grid, Z_grid])

        # Define future time factors and create animation frames
        factors = [1.0, 2.0, 6.0, 12.0, 24.0, 48.0, 100.0]
        frames = []
        slider_steps = []
        for factor in factors:
            future_time = factor / 5.0
            future_time_array = np.full((grid_points.shape[0], 1), future_time)
            grid_points_4d = np.hstack([grid_points, future_time_array])
            pred_norm = knn_model.predict(grid_points_4d)
            pred_percent = np.clip(pred_norm, 0, 1) * 100

            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=X_grid * 100,
                    y=Y_grid * 100,
                    z=Z_grid * 100,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=pred_percent,
                        colorscale='Rainbow',
                        cmin=0,
                        cmax=100,
                        colorbar=dict(title="Predicted Win Rate (%)")
                    )
                )],
                name=f"{int(factor * 100)}%"
            ))
            slider_steps.append({
                "args": [[f"{int(factor * 100)}%"],
                         {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": f"{int(factor * 100)}%",
                "method": "animate"
            })

        # Initialize figure with first frame
        init_frame = frames[0].data[0]
        fig5 = go.Figure(
            data=[init_frame],
            layout=go.Layout(
                title=f"Figure 5: Prediction Outcome using KNN at Future Time {int(factors[0]*100)}%",
                scene=dict(
                    xaxis_title="[z]:Ave MCTS Usage (%)",
                    yaxis_title="[y]:Ave DQN Usage (%)",
                    zaxis_title="[x]:Ave HYBRID Usage (%)"
                ),
                template="plotly_dark",
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Future Time: "},
                    "pad": {"t": 50},
                    "steps": slider_steps
                }]
            ),
            frames=frames
        )
        print("Displaying Figure 5 in offline mode...")
        pyo.plot(fig5, filename=os.path.join(plots_dir, "figure5.html"), auto_open=True)
        print("Figure 5 loaded successfully.")
    except Exception as e:
        print("An error occurred while plotting Figure 5:", str(e))


# --------------------- Main Execution ---------------------
winners, rewards, turns, epsilons, mcts_levels, mcts_used_rate, dqn_rates, hybrid_rates, min_reward, max_reward = parse_log_file(log_file_path)

if len(winners) == 0:
    print("No valid data parsed. Please check your log file format.")
else:
    total_games = len(winners)
    total_p2_wins = sum(1 for w in winners if w == 2)
    draws = winners.count(-1)
    
    # For Figure 1, use fixed interval = 100
    agg_data_fig1 = aggregate_data(winners, rewards, turns, mcts_used_rate, dqn_rates, hybrid_rates, FIG1_INTERVAL)
    print("Aggregated data points for Figure 1:", len(agg_data_fig1[0]))
    plot_figure1(agg_data_fig1, total_games, total_p2_wins, draws, min_reward, max_reward, FIG1_INTERVAL)
    
    # For Figure 2, use fixed interval = 1000
    agg_data_fig2 = aggregate_data(winners, rewards, turns, mcts_used_rate, dqn_rates, hybrid_rates, FIG2_INTERVAL)
    plot_figure2(agg_data_fig2[0], agg_data_fig2[1], mcts_levels, epsilons)
    
    # For Figures 3 and 4, use aggregated data with interval = 1000 and scale game index by 1/1000
    agg_data_fig3 = aggregate_data(winners, rewards, turns, mcts_used_rate, dqn_rates, hybrid_rates, FIG2_INTERVAL)
    game_index_fig3 = np.array(agg_data_fig3[0][1:]) / 1000.0
    avg_mcts_fig3 = np.array(agg_data_fig3[4][1:])
    avg_dqn_fig3 = np.array(agg_data_fig3[5][1:])
    avg_hybrid_fig3 = np.array(agg_data_fig3[6][1:])
    avg_winrates_fig3 = np.array(agg_data_fig3[1][1:])
    plot_figure3(avg_mcts_fig3, avg_dqn_fig3, avg_hybrid_fig3, avg_winrates_fig3, game_index_fig3)
    plot_figure4(agg_data_fig3, total_games, FIG2_INTERVAL)
    
    # For Figure 5, use raw row data
    plot_figure5(mcts_used_rate, dqn_rates, hybrid_rates, winners)
