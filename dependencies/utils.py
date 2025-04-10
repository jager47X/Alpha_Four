import logging
import os
import math
import numpy as np
from numba import cuda, int32, float32
import warnings
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
    Sets up a logger that logs both to a file (log_file) and the console (stdout).
    Ensures the directory for log_file exists before creating the FileHandler.
    
    Args:
        log_file (str): The path to the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicated logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure the directory for log_file exists
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            safe_make_dir(log_dir)

    # Create a file handler if log_file is provided
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # Always add a stream handler (for console logs)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    return logger


def get_next_index(path: str) -> int:
    """
    Reads the directories in 'path' and returns the next available index as an integer.
    If no directories exist, returns 0.
    
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

@cuda.jit
def simulate_games_kernel(board_states, current_players, results, num_simulations, seeds, flag, q_bias, q_threshold):
    """
    CUDA kernel to simulate a number of Connect4 games in parallel.
    The move selection uses the provided q_bias vector to deterministically choose a move
    if any valid move's bias meets or exceeds q_threshold; otherwise, a weighted random selection is done.
    
    Parameters:
        board_states : 3D int32 array (num_simulations, ROWS, COLUMNS)
        current_players : 1D int32 array (num_simulations,)
        results : 1D int32 output array (num_simulations,)
        num_simulations : total number of simulations
        seeds : 1D uint32 array used for PRNG for each simulation
        flag : 1D int32 array (used to flag kernel execution)
        q_bias : 1D float32 array of shape (COLUMNS,) with Q-values for each column from the root state
        q_threshold : float32 value that must be met for a move to be deterministically selected
    """
    idx = cuda.grid(1)
    if idx >= num_simulations:
        return

    if idx == 0:
        flag[0] = 1

    # Initialize PRNG with a unique seed per simulation.
    seed = seeds[idx]
    a = 1664525
    c = 1013904223
    m = 2**32

    # Copy board into local memory.
    board_sim = cuda.local.array((ROWS, COLUMNS), dtype=int32)
    for row in range(ROWS):
        for col in range(COLUMNS):
            board_sim[row, col] = board_states[idx, row, col]
    
    player = current_players[idx]
    winner = EMPTY
    depth = 0
    max_depth = ROWS * COLUMNS

    while depth < max_depth:
        depth += 1
        # Build list of valid moves and record corresponding biases.
        valid = cuda.local.array(7, dtype=int32)
        biases = cuda.local.array(7, dtype=float32)
        valid_count = 0
        for col in range(COLUMNS):
            if board_sim[0, col] == EMPTY:
                valid[valid_count] = col
                biases[valid_count] = q_bias[col]  # Use provided Q-bias value.
                valid_count += 1

        if valid_count == 0:
            break  # Draw – no valid move.

        # Check if any valid move has bias >= q_threshold.
        best_idx = -1
        best_bias = -1.0
        for i in range(valid_count):
            if biases[i] >= q_threshold and biases[i] > best_bias:
                best_bias = biases[i]
                best_idx = i

        if best_idx != -1:
            # Deterministically choose this move.
            action = valid[best_idx]
        else:
            # Weighted random selection: sum biases over valid moves.
            total_bias = 0.0
            for i in range(valid_count):
                total_bias += biases[i]
            if total_bias > 0:
                rand_num = (a * seed + c) % m
                seed = rand_num
                threshold = ((rand_num % 1000) / 1000.0) * total_bias
                cum_sum = 0.0
                selected_action = valid[0]
                for i in range(valid_count):
                    cum_sum += biases[i]
                    if cum_sum >= threshold:
                        selected_action = valid[i]
                        break
                action = selected_action
            else:
                # Fallback to uniform random selection.
                rand_num = (a * seed + c) % m
                seed = rand_num
                action = valid[rand_num % valid_count]

        # Execute move: drop piece into first available row from bottom.
        for row in range(ROWS - 1, -1, -1):
            if board_sim[row, action] == EMPTY:
                board_sim[row, action] = player
                break

        # Check for win condition.
        win = False
        # Horizontal check.
        for r in range(ROWS):
            count = 1
            last = board_sim[r, 0]
            for c in range(1, COLUMNS):
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

        # Vertical check.
        if not win:
            for c in range(COLUMNS):
                count = 1
                last = board_sim[0, c]
                for r in range(1, ROWS):
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

        # Diagonal Down-Right check.
        if not win:
            for r in range(ROWS - WIN_LENGTH + 1):
                for c in range(COLUMNS - WIN_LENGTH + 1):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, WIN_LENGTH):
                        if board_sim[r + i, c + i] != first:
                            match = False
                            break
                    if match:
                        winner = first
                        win = True
                        break
                if win:
                    break

        # Diagonal Up-Right check.
        if not win:
            for r in range(WIN_LENGTH - 1, ROWS):
                for c in range(COLUMNS - WIN_LENGTH + 1):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, WIN_LENGTH):
                        if board_sim[r - i, c + i] != first:
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

        # If top row is full then it's a draw.
        draw = True
        for c in range(COLUMNS):
            if board_sim[0, c] == EMPTY:
                draw = False
                break
        if draw:
            break

        # Switch player.
        player = 3 - player

    results[idx] = winner

def prepare_simulation_data(env: 'Connect4', num_simulations: int):
    """
    Prepare data for CUDA-based simulation.
    
    Returns:
        board_states: (num_simulations, ROWS, COLUMNS) int32 array
        current_players: (num_simulations,) int32 array
        results: (num_simulations,) int32 zeros array
        seeds: (num_simulations,) uint32 array for PRNG seeds.
    """
    board = env.get_board()  # Expected shape (ROWS, COLUMNS)
    board_states = np.tile(board, (num_simulations, 1, 1)).astype(np.int32)
    current_players = np.full(num_simulations, env.current_player, dtype=np.int32)
    results = np.zeros(num_simulations, dtype=np.int32)
    seeds = np.random.randint(0, 2**32, size=num_simulations, dtype=np.uint32)
    return board_states, current_players, results, seeds

def run_simulations_cuda(env: 'Connect4', num_simulations: int = 4096, block_size: int = 256,
                         q_bias=None, q_threshold=0.5):
    """
    Run Connect4 game simulations on the GPU using CUDA.
    
    If q_bias is provided (a 1D array of Q-values for each column) and moves meet or exceed
    q_threshold, those moves will be deterministically selected in the simulation.
    """
    board_states, current_players, _, seeds = prepare_simulation_data(env, num_simulations)
    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds)
    d_results = cuda.device_array(num_simulations, dtype=np.int32)
    grid_size = math.ceil(num_simulations / block_size)
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)

    if q_bias is None:
        q_bias = np.zeros(COLUMNS, dtype=np.float32)
    d_q_bias = cuda.to_device(q_bias)

    try:
        simulate_games_kernel[grid_size, block_size](d_board_states, d_current_players, d_results,
                                                      num_simulations, d_seeds, d_flag, d_q_bias,
                                                      q_threshold)
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
