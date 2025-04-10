import logging
import os
import math
import numpy as np
from numba import cuda
import numba
import warnings
import random
import torch
import time
from numba.core.errors import NumbaPerformanceWarning
from .environment import Connect4

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ----------------- Utility Functions ----------------- #
def safe_make_dir(directory: str) -> None:
    """
    Creates the specified directory if it doesn't exist.
    
    Args:
        directory (str): The path of the directory to create.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")
        raise

def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that logs both to a file (log_file) and the console.
    
    Args:
        log_file (str): The path to the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            safe_make_dir(log_dir)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
    return logger

def get_next_index(path: str) -> int:
    """
    Reads the directories in 'path' and returns the next available index.
    
    Args:
        path (str): The directory path to search.
    
    Returns:
        int: The next available index.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return 0
    existing_dirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.isdigit()
    ]
    existing_indices = [int(d) for d in existing_dirs]
    next_index = max(existing_indices) + 1 if existing_indices else 0
    return next_index

# ----------------- CUDA Simulation Functions ----------------- #
# Constants for Connect4
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4

import math
import numpy as np
from numba import cuda
import numba
import random
import torch
from .environment import Connect4

# Constants for Connect4
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4

@cuda.jit
def simulate_games_kernel(board_states, current_players, results, num_simulations, seeds, flag, q_bias, q_threshold):
    """
    CUDA kernel to simulate a number of Connect4 games in parallel.
    For each simulation, if any valid move has a q_bias value that meets or exceeds q_threshold,
    that move is chosen deterministically; otherwise, the move is selected via weighted random choice.
    """
    idx = cuda.grid(1)
    if idx >= num_simulations:
        return

    if idx == 0:
        flag[0] = 1

    # Initialize PRNG using a simple LCG.
    seed = seeds[idx]
    a = 1664525
    c = 1013904223
    m = 2**32

    # Copy board state into shared local memory.
    board_sim = cuda.local.array((6, 7), dtype=numba.int32)
    for row in range(6):
        for col in range(7):
            board_sim[row, col] = board_states[idx, row, col]

    player = current_players[idx]
    winner = EMPTY
    depth = 0
    max_depth = ROWS * COLUMNS

    while depth < max_depth:
        depth += 1
        # Build list of valid actions.
        valid = cuda.local.array(7, dtype=numba.int32)
        valid_count = 0
        for col in range(7):
            if board_sim[0, col] == EMPTY:
                valid[valid_count] = col
                valid_count += 1

        if valid_count == 0:
            break  # Draw

        # Check if any valid move has a q_bias >= q_threshold.
        best_idx = -1
        best_bias = -1.0
        for i in range(valid_count):
            move = valid[i]
            bias_val = q_bias[move]
            if bias_val >= q_threshold and bias_val > best_bias:
                best_bias = bias_val
                best_idx = i

        if best_idx != -1:
            action = valid[best_idx]
        else:
            # Fall back to weighted random selection using q_bias as weights.
            total_bias = 0.0
            for i in range(valid_count):
                total_bias += q_bias[ valid[i] ]
            if total_bias > 0:
                rand_num = (a * seed + c) % m
                seed = rand_num
                threshold = ((rand_num % 1000) / 1000.0) * total_bias
                cum_sum = 0.0
                selected_action = valid[0]
                for i in range(valid_count):
                    cum_sum += q_bias[ valid[i] ]
                    if cum_sum >= threshold:
                        selected_action = valid[i]
                        break
                action = selected_action
            else:
                # Fallback to uniform random selection.
                rand_num = (a * seed + c) % m
                seed = rand_num
                action = valid[rand_num % valid_count]

        # Make move: drop the piece in the lowest available row.
        for row in range(ROWS - 1, -1, -1):
            if board_sim[row, action] == EMPTY:
                board_sim[row, action] = player
                break

        # Check win conditions (horizontal, vertical, and two diagonal checks).
        win = False
        # Horizontal Check.
        for r in range(6):
            count = 1
            last = board_sim[r, 0]
            for c in range(1, 7):
                if board_sim[r, c] == last and board_sim[r, c] != EMPTY:
                    count += 1
                    if count >= WIN_LENGTH:
                        winner = last
                        win = True
                        break
                else:
                    last = board_sim[r, c]
                    count = 1
            if win:
                break

        if not win:
            # Vertical Check.
            for c in range(7):
                count = 1
                last = board_sim[0, c]
                for r in range(1, 6):
                    if board_sim[r, c] == last and board_sim[r, c] != EMPTY:
                        count += 1
                        if count >= WIN_LENGTH:
                            winner = last
                            win = True
                            break
                    else:
                        last = board_sim[r, c]
                        count = 1
                if win:
                    break

        if not win:
            # Diagonal Down-Right Check.
            for r in range(3):
                for c in range(4):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, 4):
                        if board_sim[r+i, c+i] != first:
                            match = False
                            break
                    if match:
                        winner = first
                        win = True
                        break
                if win:
                    break

        if not win:
            # Diagonal Up-Right Check.
            for r in range(3, 6):
                for c in range(4):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, 4):
                        if board_sim[r-i, c+i] != first:
                            match = False
                            break
                    if match:
                        winner = first
                        win = True
                        break
                if win:
                    break

        if win:
            break

        # Check for draw condition: if top row is full.
        draw = True
        for c in range(7):
            if board_sim[0, c] == EMPTY:
                draw = False
                break
        if draw:
            break

        # Toggle player.
        player = 3 - player

    results[idx] = winner

def prepare_simulation_data(env: Connect4, num_simulations: int):
    """
    Prepares simulation data for running CUDA-based simulations.
    
    Args:
        env (Connect4): The current Connect4 environment.
        num_simulations (int): The number of simulations to run.
        
    Returns:
        tuple: (board_states, current_players, results, seeds)
    """
    board = env.get_board()  # Expected shape: (ROWS, COLUMNS)
    board_states = np.tile(board, (num_simulations, 1, 1)).astype(np.int32)
    current_players = np.full(num_simulations, env.current_player, dtype=np.int32)
    results = np.zeros(num_simulations, dtype=np.int32)
    seeds = np.random.randint(0, 2**32, size=num_simulations, dtype=np.uint32)
    return board_states, current_players, results, seeds

def run_simulations_cuda(env: Connect4, num_simulations: int = 4096, block_size: int = 256,
                         dqn_model=None, hybrid: bool = False, q_threshold: float = 0.5, q_bias=None):
    """
    Runs a number of Connect4 simulations on the GPU using CUDA.
    
    Modes:
      - Hybrid mode (if dqn_model is provided and hybrid == True):
          If no q_bias is provided, it should be computed by the caller.
          The provided q_bias vector and q_threshold are passed to the kernel so that
          moves with a bias above the threshold are selected deterministically.
      - Normal mode: Uses random rollouts (q_bias remains a zero vector).
    
    Args:
        env (Connect4): The current Connect4 environment.
        num_simulations (int): Number of simulations to run.
        block_size (int): CUDA block size.
        dqn_model: A PyTorch DQN model (not used here if q_bias is already provided).
        hybrid (bool): Whether to use hybrid mode.
        q_threshold (float): Threshold for deterministic move selection.
        q_bias (np.ndarray or None): A precomputed Q-bias vector (length = COLUMNS). If None, a zero vector is used.
        
    Returns:
        np.ndarray or None: Array of simulation results (winning player IDs) or None on error.
    """
    if q_bias is None:
        q_bias = np.zeros(COLUMNS, dtype=np.float32)

    board_states, current_players, _, seeds = prepare_simulation_data(env, num_simulations)
    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds)
    d_results = cuda.device_array(num_simulations, dtype=np.int32)
    grid_size = math.ceil(num_simulations / block_size)
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)
    d_q_bias = cuda.to_device(q_bias)
    q_threshold_device = np.float32(q_threshold)

    try:
        simulate_games_kernel[grid_size, block_size](d_board_states, d_current_players, d_results,
                                                      num_simulations, d_seeds, d_flag,
                                                      d_q_bias, q_threshold_device)
        cuda.synchronize()
    except cuda.CudaSupportError as e:
        print(f"CUDA Support Error: {e}")
        return None
    except cuda.CudaAPIError as e:
        print(f"CUDA API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    results = d_results.copy_to_host()
    return results
