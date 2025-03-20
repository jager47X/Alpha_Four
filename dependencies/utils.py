import logging
import os
import math
import numpy as np
from numba import cuda
import numba
import warnings
from numba.core.errors import NumbaPerformanceWarning

from .replay_buffer import DiskReplayBuffer
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
def simulate_games_kernel(board_states, current_players, results, num_simulations, seeds, flag):
    """
    CUDA kernel to simulate a number of Connect4 games in parallel.
    
    Each thread simulates one game (starting from a given board state) until a win, draw, or full board.
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
        # Build list of valid actions
        valid = cuda.local.array(7, dtype=numba.int32)
        valid_count = 0
        for col in range(7):
            if board_sim[0, col] == EMPTY:
                if valid_count < 7:
                    valid[valid_count] = col
                    valid_count += 1
        if valid_count == 0:
            break  # Draw

        # Generate a pseudo-random number using LCG
        rand_num = (a * seed + c) % m
        seed = rand_num
        action = valid[rand_num % valid_count]

        # Make the move: place the piece in the first available row from the bottom
        for row in range(5, -1, -1):
            if board_sim[row, action] == EMPTY:
                board_sim[row, action] = player
                break

        # Check for a win condition
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

        # Check for draw: if top row is full, then it's a draw.
        draw = True
        for c in range(7):
            if board_sim[0, c] == EMPTY:
                draw = False
                break
        if draw:
            break

        # Switch player for next move (toggle between 1 and 2)
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


def run_simulations_cuda(env: Connect4, num_simulations: int = 4096, block_size: int = 256):
    """
    Runs a number of Connect4 game simulations on the GPU using CUDA.
    
    Args:
        env (Connect4): The Connect4 environment to simulate from.
        num_simulations (int, optional): The number of simulations to run. Defaults to 4096.
        block_size (int, optional): CUDA block size. Defaults to 256.
    
    Returns:
        np.ndarray or None: An array of simulation results (winning player for each simulation)
                            or None if an error occurred.
    """
    board_states, current_players, _, seeds = prepare_simulation_data(env, num_simulations)
    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds.astype(np.uint32))
    d_results = cuda.device_array(num_simulations, dtype=np.int32)
    grid_size = math.ceil(num_simulations / block_size)
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)
    try:
        simulate_games_kernel[grid_size, block_size](d_board_states, d_current_players, d_results, num_simulations, d_seeds, d_flag)
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
