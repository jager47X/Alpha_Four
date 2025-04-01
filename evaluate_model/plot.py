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
    """
    Plots three subplots (MCTS, DQN, HYBRID vs. Ave Win Rate) and allows the user
    to filter by 10 equal sections of the scaled game index (0–10%, 10–20%, etc.).

    Parameters:
    -----------
    avg_mcts_usage : list or array
        Average MCTS usage per game (0 to 100 range).
    avg_dqn_usage : list or array
        Average DQN usage per game (0 to 100 range).
    avg_hybrid_usage : list or array
        Average HYBRID usage per game (0 to 100 range).
    avg_winrates : list or array
        Average win rate per game (0 to 100 range).
    game_index : list or array
        The raw game indices (e.g., 0, 1, 2, ...). Will be scaled by 1/1000 here.
    """
    try:
        print("Starting Figure 3 plotting with 10 equal sections of the scaled game index...")

        import numpy as np
        import os
        import plotly.graph_objects as go
        import plotly.offline as pyo
        from plotly.subplots import make_subplots

        # Helper function to compute a linear trend line unless x has no variation
        def compute_trend_line(x, y, threshold=1e-8):
            if len(x) < 2 or np.std(x) < threshold:
                # If there's not enough variation, return a flat trend line at the mean of y
                if len(x) > 0:
                    x_line = np.linspace(np.min(x), np.max(x), 100)
                    y_line = np.full_like(x_line, np.mean(y))
                else:
                    x_line, y_line = [], []
                return x_line, y_line

            coeffs = np.polyfit(x, y, 1)  # 1 -> linear
            poly = np.poly1d(coeffs)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = poly(x_line)
            return x_line, y_line

        # Scale the game index by 1/1000
        game_ix = np.array(game_index) / 1000.0
        mcts = np.array(avg_mcts_usage)
        dqn = np.array(avg_dqn_usage)
        hybrid = np.array(avg_hybrid_usage)
        winrates = np.array(avg_winrates)

        # Determine the min and max of the scaled game index
        min_game, max_game = game_ix.min(), game_ix.max()

        # Prepare dictionary to store data for each of the 10 bins
        filter_options = {}

        # Create 10 bins from 0-10%, 10-20%, ... 90-100% of the scaled range
        for i in range(10):
            lower = min_game + (max_game - min_game) * (i / 10)
            upper = min_game + (max_game - min_game) * ((i + 1) / 10)

            # Label for the slider (e.g., "0-10%", "10-20%", etc.)
            label = f"{i * 10}-{(i + 1) * 10}%"

            # Select data points whose scaled game index is in [lower, upper]
            mask = (game_ix >= lower) & (game_ix <= upper)
            indices = np.where(mask)[0]

            if indices.size == 0:
                # If no data in this bin, keep everything empty
                filter_options[label] = {
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
                # Subset for this bin
                f_mcts = mcts[indices]
                f_dqn = dqn[indices]
                f_hybrid = hybrid[indices]
                f_winrates = winrates[indices]

                # Compute trend lines
                filter_options[label] = {
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

        # Create the subplots for MCTS, DQN, and HYBRID
        fig3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=("MCTS vs. Avg Win Rate", "DQN vs. Avg Win Rate", "HYBRID vs. Avg Win Rate")
        )

        # We'll initialize the figure with the first bin: "0-10%"
        first_bin_label = "0-10%"
        init_data = filter_options[first_bin_label]

        # --- MCTS Subplot ---
        fig3.add_trace(go.Scatter(
            x=init_data["mcts_x"],
            y=init_data["mcts_y"],
            mode='markers',
            marker=dict(color='red'),
            name="MCTS Usage"
        ), row=1, col=1)

        mcts_trend_x, mcts_trend_y = init_data["mcts_trend"]
        fig3.add_trace(go.Scatter(
            x=mcts_trend_x,
            y=mcts_trend_y,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="Trend (MCTS)"
        ), row=1, col=1)

        # --- DQN Subplot ---
        fig3.add_trace(go.Scatter(
            x=init_data["dqn_x"],
            y=init_data["dqn_y"],
            mode='markers',
            marker=dict(color='magenta'),
            name="DQN Usage"
        ), row=1, col=2)

        dqn_trend_x, dqn_trend_y = init_data["dqn_trend"]
        fig3.add_trace(go.Scatter(
            x=dqn_trend_x,
            y=dqn_trend_y,
            mode='lines',
            line=dict(color='magenta', dash='dash'),
            name="Trend (DQN)"
        ), row=1, col=2)

        # --- HYBRID Subplot ---
        fig3.add_trace(go.Scatter(
            x=init_data["hybrid_x"],
            y=init_data["hybrid_y"],
            mode='markers',
            marker=dict(color='orange'),
            name="HYBRID Usage"
        ), row=1, col=3)

        hybrid_trend_x, hybrid_trend_y = init_data["hybrid_trend"]
        fig3.add_trace(go.Scatter(
            x=hybrid_trend_x,
            y=hybrid_trend_y,
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name="Trend (HYBRID)"
        ), row=1, col=3)

        # Set axis titles
        for col in [1, 2, 3]:
            fig3.update_xaxes(title_text="Strategy Usage (%)", row=1, col=col)
            fig3.update_yaxes(title_text="Avg Win Rate (%)", row=1, col=col)

        # Main layout
        fig3.update_layout(
            title_text="Figure 3: Strategy Usage vs. Win Rate (Divided into 10% Scaled Bins)",
            template="plotly_dark"
        )

        # Build the slider steps
        slider_steps = []
        for label, data in filter_options.items():
            step = {
                "args": [{
                    "x": [
                        data["mcts_x"], data["mcts_trend"][0],
                        data["dqn_x"], data["dqn_trend"][0],
                        data["hybrid_x"], data["hybrid_trend"][0]
                    ],
                    "y": [
                        data["mcts_y"], data["mcts_trend"][1],
                        data["dqn_y"], data["dqn_trend"][1],
                        data["hybrid_y"], data["hybrid_trend"][1]
                    ]
                }],
                "label": label,
                "method": "restyle"
            }
            slider_steps.append(step)

        # Add a single slider with the 10 bins
        fig3.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Scaled Game Index Range: "},
                "pad": {"t": 50},
                "x": 0.5,
                "xanchor": "center",
                "y": 0,
                "yanchor": "top",
                "steps": slider_steps
            }]
        )

        print("Displaying Figure 3 in offline mode...")
        plots_dir = "."  # Adjust directory as needed
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
                xaxis_title="[x]Ave MCTS Usage (%)",
                yaxis_title="[y]Ave DQN Usage (%)",
                zaxis_title="[z]Ave HYBRID Usage (%)"
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

def plot_figure5(avg_mcts_usage, avg_dqn_usage, avg_hybrid_usage, winners,game_index):
    try:
        print("Starting Figure 5 plotting (prediction outcome with slider)...")
        import numpy as np
        import os
        import plotly.graph_objects as go
        import plotly.offline as pyo
        from sklearn.neighbors import KNeighborsRegressor

        # Normalize inputs (assumed percentages)
        X_mcts = np.array(avg_mcts_usage) / 100.0
        X_dqn = np.array(avg_dqn_usage) / 100.0
        X_hybrid = np.array(avg_hybrid_usage) / 100.0

        # Compute binary wins (1 for win, 0 for loss) for the rolling window calculation.
        binary_wins = np.where(np.array(winners) == 2, 1.0, 0.0)

        # Create a normalized time array (scaled by 1/5)
        n = len(X_mcts)
        time_array = np.linspace(0, 1, n) / 5.0

        # Compute opponent strength over time using a 100-game rolling window.
        # Start at 20 and, for each window with >= 80% win rate, increase by 20 (up to a cap of 2000).
        opponent_strength_raw = []
        current_strength = 20
        for i in range(n):
            if i >= 99:
                window_win_rate = np.mean(binary_wins[i-99:i+1])
                if window_win_rate >= 0.8:
                    current_strength = min(2000, current_strength + 20)
            opponent_strength_raw.append(current_strength)
        opponent_strength_raw = np.array(opponent_strength_raw)
        # Normalize opponent strength to [0, 1] by dividing by 2000
        norm_opponent_strength = opponent_strength_raw / 2000.0

        # Adjust the training target so that a win is 1.0, but a loss is given a value
        # that increases with opponent strength (i.e. reducing the loss penalty).
        # For instance, if a game is lost, we assign 0.2 * norm_opponent_strength.
        y_winrates = np.where(np.array(winners) == 2, 1.0, 0.2 * norm_opponent_strength)

        # Enforce MCTS + DQN + HYBRID = 1.0 exactly
        total = X_mcts + X_dqn + X_hybrid
        X_mcts /= total
        X_dqn /= total
        X_hybrid /= total

        # Combine features: [MCTS, DQN, HYBRID, time, opponent_strength]
        features = np.column_stack([X_mcts, X_dqn, X_hybrid, time_array, norm_opponent_strength])

        # Set the n_neibors based on the number of game index
        n_neighbors_dynamic = max(3, min(int(game_index / 100000), 10))

        # Train the KNN model with the adjusted data
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors_dynamic)
        knn_model.fit(features, y_winrates)
        print("KNN model fitted on augmented data including opponent strength and adjusted loss penalty.")

        # Create prediction grid for usage percentages where MCTS + DQN + HYBRID = 1.0
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

        # For future prediction, update time and opponent strength.
        # Use the last known normalized opponent strength as baseline.
        baseline_strength = norm_opponent_strength[-1]
        # Future factors define our prediction steps.
        factors = [1.0, 2.0, 6.0, 12.0, 24.0, 48.0, 100.0]
        frames = []
        slider_steps = []
        for factor in factors:
            future_time = factor / 5.0
            # Increase opponent strength with future factor.
            future_strength = min(1.0, baseline_strength + 0.01 * factor)
            future_time_array = np.full((grid_points.shape[0], 1), future_time)
            opponent_strength_array = np.full((grid_points.shape[0], 1), future_strength)
            # New 5D feature vector for prediction.
            grid_points_5d = np.hstack([grid_points, future_time_array, opponent_strength_array])
            pred_norm = knn_model.predict(grid_points_5d)
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
                name=f"Time {int(factor*100)}%"
            ))
            slider_steps.append({
                "args": [[f"Time {int(factor*100)}%"],
                         {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": f"{int(factor*100)}%",
                "method": "animate"
            })

        # Initialize figure with the first frame
        init_frame = frames[0].data[0]
        fig5 = go.Figure(
            data=[init_frame],
            layout=go.Layout(
                title=f"Figure 5: Prediction Outcome with Future Time {int(factors[0]*100)}%",
                scene=dict(
                    xaxis_title="[x]: Ave MCTS Usage (%)",
                    yaxis_title="[y]: Ave DQN Usage (%)",
                    zaxis_title="[z]: Ave HYBRID Usage (%)"
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
        plots_dir = "."  # Adjust as needed
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
    plot_figure5(mcts_used_rate, dqn_rates, hybrid_rates, winners,game_index=total_games)
