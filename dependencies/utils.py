import logging
import os
import sys
import math
import numpy as np

# Fix Numba CUDA nvvm discovery for CUDA 13.x (nvvm dll moved to nvvm/bin/x64/)
if sys.platform == 'win32':
    _cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH', '')
    _nvvm_x64 = os.path.join(_cuda_home, 'nvvm', 'bin', 'x64')
    if os.path.isdir(_nvvm_x64):
        try:
            os.add_dll_directory(_nvvm_x64)
        except (OSError, AttributeError):
            pass
        # Patch numba's nvvm path resolution to include x64 subdir
        try:
            import numba.cuda.cuda_paths as _cp
            _orig_nvvm_lib_dir = _cp._nvvm_lib_dir
            def _patched_nvvm_lib_dir():
                return 'nvvm', 'bin', 'x64'
            _cp._nvvm_lib_dir = _patched_nvvm_lib_dir
        except Exception:
            pass

from numba import cuda
import numba
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
def simulate_games_kernel(board_states, current_players, results, num_simulations, seeds, flag, q_bias):
    """
    CUDA kernel to simulate a number of Connect4 games in parallel,
    using a bias based on externally provided Q-values for move selection.
    
    q_bias: a 1D array of shape (COLUMNS,) with Q-values for each column.
    """
    idx = cuda.grid(1)
    if idx >= num_simulations:
        return

    # Indicate that at least one thread is executing
    if idx == 0:
        flag[0] = 1

    # Initialize PRNG with a unique seed per simulation
    seed = seeds[idx]
    a = 1664525
    c = 1013904223
    m = 2**32

    # Each thread gets its own local board copy
    board_sim = cuda.local.array((6, 7), dtype=numba.int32)
    for row in range(6):
        for col in range(7):
            board_sim[row, col] = board_states[idx, row, col]
    player = current_players[idx]
    winner = EMPTY
    depth = 0
    max_depth = ROWS * COLUMNS  # Maximum moves possible

    while depth < max_depth:
        depth += 1
        # Build list of valid actions and collect Q-value biases for these actions.
        valid = cuda.local.array(7, dtype=numba.int32)
        biases = cuda.local.array(7, dtype=numba.float32)
        valid_count = 0
        total_bias = 0.0
        for col in range(7):
            if board_sim[0, col] == EMPTY:
                valid[valid_count] = col
                # Use the externally provided Q-value bias for this column.
                bias_val = q_bias[col]
                biases[valid_count] = bias_val
                total_bias += bias_val
                valid_count += 1
        if valid_count == 0:
            break  # Draw

        # Choose action: if total_bias > 0, use weighted selection; otherwise, fall back to uniform random.
        if total_bias > 0:
            # Generate a pseudo-random number for weighted selection.
            rand_num = (a * seed + c) % m
            seed = rand_num
            # Map random number to a float in [0, total_bias)
            threshold = ((rand_num % 1000) / 1000.0) * total_bias
            cum_sum = 0.0
            selected_action = valid[0]  # default action
            for i in range(valid_count):
                cum_sum += biases[i]
                if cum_sum >= threshold:
                    selected_action = valid[i]
                    break
            action = selected_action
        else:
            # Fallback: uniform random selection
            rand_num = (a * seed + c) % m
            seed = rand_num
            action = valid[rand_num % valid_count]

        # Make the move: place the piece in the first available row from the bottom
        for row in range(5, -1, -1):
            if board_sim[row, action] == EMPTY:
                board_sim[row, action] = player
                break

        # Check for a win condition (horizontal, vertical, diagonal checks)
        win = False
        # Horizontal Check
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

        # Vertical Check
        if not win:
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

        # Diagonal Down-Right Check
        if not win:
            for r in range(3):
                for c in range(4):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, 4):
                        if board_sim[r + i, c + i] != first:
                            match = False
                            break
                    if match:
                        winner = first
                        win = True
                        break
                if win:
                    break

        # Diagonal Up-Right Check
        if not win:
            for r in range(3, 6):
                for c in range(4):
                    first = board_sim[r, c]
                    if first == EMPTY:
                        continue
                    match = True
                    for i in range(1, 4):
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

        # Check for draw: if the top row is full, then it's a draw.
        draw = True
        for c in range(7):
            if board_sim[0, c] == EMPTY:
                draw = False
                break
        if draw:
            break

        # Switch player for next move
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
    board = env.get_board()
    board_states = np.tile(board, (num_simulations, 1, 1)).astype(np.int32)
    current_players = np.full(num_simulations, env.current_player, dtype=np.int32)
    results = np.zeros(num_simulations, dtype=np.int32)
    seeds = np.random.randint(0, 2**32, size=num_simulations, dtype=np.uint32)
    return board_states, current_players, results, seeds

def run_simulations_cuda(env: Connect4, num_simulations: int = 4096, block_size: int = 256, q_bias=None):
    """
    Runs a number of Connect4 game simulations on the GPU using CUDA.
    
    If q_bias is provided (a 1D array of Q-values for each column), it will be used
    to bias the random move selection during simulations.
    """
    board_states, current_players, _, seeds = prepare_simulation_data(env, num_simulations)
    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds.astype(np.uint32))
    d_results = cuda.device_array(num_simulations, dtype=np.int32)
    grid_size = math.ceil(num_simulations / block_size)
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)

    # If no q_bias is provided, use a zero array so that the kernel falls back to uniform selection.
    if q_bias is None:
        q_bias = np.zeros(COLUMNS, dtype=np.float32)
    d_q_bias = cuda.to_device(q_bias)

    try:
        simulate_games_kernel[grid_size, block_size](d_board_states, d_current_players, d_results,
                                                      num_simulations, d_seeds, d_flag, d_q_bias)
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


def _simulate_one_game_cpu(board, current_player, q_bias, rng):
    """CPU fallback: simulate one random Connect4 game to completion."""
    b = board.copy()
    player = current_player
    rows, cols = b.shape

    for _ in range(rows * cols):
        valid = [c for c in range(cols) if b[0, c] == 0]
        if not valid:
            return 0  # draw

        # Weighted random selection using q_bias
        if q_bias is not None and np.any(q_bias != 0):
            weights = np.array([max(q_bias[c], 0.01) for c in valid])
            weights /= weights.sum()
            col = rng.choice(valid, p=weights)
        else:
            col = rng.choice(valid)

        # Drop piece
        for r in range(rows - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break

        # Check winner (simplified inline check)
        for r in range(rows):
            for c in range(cols - 3):
                if b[r, c] == player and b[r, c+1] == player and b[r, c+2] == player and b[r, c+3] == player:
                    return player
        for r in range(rows - 3):
            for c in range(cols):
                if b[r, c] == player and b[r+1, c] == player and b[r+2, c] == player and b[r+3, c] == player:
                    return player
        for r in range(rows - 3):
            for c in range(cols - 3):
                if b[r, c] == player and b[r+1, c+1] == player and b[r+2, c+2] == player and b[r+3, c+3] == player:
                    return player
        for r in range(3, rows):
            for c in range(cols - 3):
                if b[r, c] == player and b[r-1, c+1] == player and b[r-2, c+2] == player and b[r-3, c+3] == player:
                    return player

        player = 3 - player

    return 0  # draw


def run_simulations_cpu(env, num_simulations=4096, q_bias=None):
    """CPU fallback for run_simulations_cuda when CUDA toolkit is unavailable."""
    board = env.get_board().astype(np.int32)
    results = np.zeros(num_simulations, dtype=np.int32)
    rng = np.random.default_rng()
    for i in range(num_simulations):
        results[i] = _simulate_one_game_cpu(board, env.current_player, q_bias, rng)
    return results


def run_batched_simulations_cuda(envs, sims_per_env, block_size=256, q_bias=None):
    """
    Run simulations for multiple board states in a single CUDA kernel launch.

    Args:
        envs: list of Connect4 environments (one per action).
        sims_per_env: list of int, number of simulations for each env.
        block_size: CUDA block size.
        q_bias: optional 1D array of shape (COLUMNS,) with Q-value biases.

    Returns:
        list of np.ndarray, one results array per env, or None on failure.
    """
    total_sims = sum(sims_per_env)
    if total_sims == 0:
        return [np.zeros(s, dtype=np.int32) for s in sims_per_env]

    # Build concatenated arrays for all envs
    all_boards = []
    all_players = []
    for env, n in zip(envs, sims_per_env):
        board = env.get_board().astype(np.int32)
        all_boards.append(np.tile(board, (n, 1, 1)))
        all_players.append(np.full(n, env.current_player, dtype=np.int32))

    board_states = np.concatenate(all_boards)
    current_players = np.concatenate(all_players)
    seeds = np.random.randint(0, 2**32, size=total_sims, dtype=np.uint32)

    if q_bias is None:
        q_bias = np.zeros(COLUMNS, dtype=np.float32)

    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds)
    d_results = cuda.device_array(total_sims, dtype=np.int32)
    d_q_bias = cuda.to_device(q_bias)
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)

    grid_size = math.ceil(total_sims / block_size)

    try:
        simulate_games_kernel[grid_size, block_size](
            d_board_states, d_current_players, d_results,
            total_sims, d_seeds, d_flag, d_q_bias
        )
        cuda.synchronize()
    except (cuda.CudaSupportError, cuda.CudaAPIError, Exception) as e:
        print(f"Batched CUDA simulation error: {e}")
        return None

    results = d_results.copy_to_host()

    # Split results back per-env
    split_results = []
    offset = 0
    for n in sims_per_env:
        split_results.append(results[offset:offset + n])
        offset += n
    return split_results


def run_batched_simulations_cpu(envs, sims_per_env, q_bias=None):
    """CPU fallback for run_batched_simulations_cuda."""
    rng = np.random.default_rng()
    all_results = []
    for env, n in zip(envs, sims_per_env):
        board = env.get_board().astype(np.int32)
        results = np.zeros(n, dtype=np.int32)
        for i in range(n):
            results[i] = _simulate_one_game_cpu(board, env.current_player, q_bias, rng)
        all_results.append(results)
    return all_results

