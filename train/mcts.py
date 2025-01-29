# mcts.py

import math
import numpy as np
from copy import deepcopy
import logging
from numba import cuda, int32
import numba
import random
# ----------------------------- Constants ----------------------------- #
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

ROWS = 6
COLUMNS = 7
WIN_LENGTH = 4

# ------------------------ Connect4 Environment ------------------------ #
# Assuming Connect4 is imported from environment.py
from enviroment import Connect4

# ----------------------- CUDA Simulation Kernel ----------------------- #
@cuda.jit
def simulate_games_kernel(board_states, current_players, results, num_simulations, seeds, flag):
    idx = cuda.grid(1)
    if idx >= num_simulations:
        return

    # Indicate that at least one thread is executing
    if idx == 0:
        flag[0] = 1

    # Initialize PRNG with a unique seed per simulation
    seed = seeds[idx]

    # LCG parameters (Numerical Recipes)
    a = 1664525
    c = 1013904223
    m = 2**32

    # Each thread works on one simulation
    board_sim = cuda.local.array((6, 7), dtype=numba.int32)  # Match environment.py
    # Copy the board state
    for row in range(6):
        for col in range(7):
            board_sim[row, col] = board_states[idx, row, col]
    player = current_players[idx]
    winner = EMPTY
    depth = 0
    max_depth = ROWS * COLUMNS  # Total possible moves

    while depth < max_depth:
        depth += 1
        # Initialize fixed-size array for valid actions
        valid = cuda.local.array(7, dtype=numba.int32)  # Max 7 columns
        valid_count = 0

        # Collect valid actions
        for col in range(7):
            if board_sim[0, col] == EMPTY:
                if valid_count < 7:
                    valid[valid_count] = col
                    valid_count += 1

        if valid_count == 0:
            break  # Draw

        # Generate a pseudo-random number using LCG
        rand_num = (a * seed + c) % m
        seed = rand_num  # Update seed

        # Select a random valid action using the PRNG
        action = valid[rand_num % valid_count]

        # Make move
        for row in range(5, -1, -1):
            if board_sim[row, action] == EMPTY:
                board_sim[row, action] = player
                break

        # Check for a win
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
                        if board_sim[r+i, c+i] != first:
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

        # Check for draw
        draw = True
        for c in range(7):
            if board_sim[0, c] == EMPTY:
                draw = False
                break
        if draw:
            break

        # Switch player
        player = 3 - player  # Toggle between 1 and 2

    results[idx] = winner

# ---------------------- Simulation Functions ---------------------- #
def prepare_simulation_data(env, num_simulations):
    # Create multiple copies of the board state
    board = env.get_board()
    board_states = np.tile(board, (num_simulations, 1, 1)).astype(np.int32)  # Ensured int32
    current_players = np.full(num_simulations, env.current_player, dtype=np.int32)
    results = np.zeros(num_simulations, dtype=np.int32)
    # Generate unique seeds for each simulation using uint32
    seeds = np.random.randint(0, 2**32, size=num_simulations, dtype=np.uint32)
    ##print(f"Seeds dtype: {seeds.dtype}")  # Should output: uint32
    return board_states, current_players, results, seeds

def run_simulations_cuda(env, num_simulations=4096, block_size=256):
    #print(f"Running CUDA simulations with num_simulations={num_simulations}, block_size={block_size}")
    board_states, current_players, _, seeds = prepare_simulation_data(env, num_simulations)
    #print(f"board_states shape: {board_states.shape}, dtype: {board_states.dtype}")
    # Transfer data to device
    d_board_states = cuda.to_device(board_states)
    d_current_players = cuda.to_device(current_players)
    d_seeds = cuda.to_device(seeds.astype(np.uint32))  # Ensure uint32
    #print(f"Device seeds dtype: {d_seeds.dtype}")  # Should output: uint32
    d_results = cuda.device_array(num_simulations, dtype=np.int32)
    # Define grid size
    grid_size = math.ceil(num_simulations / block_size)
    #print(f"Launching kernel with grid_size={grid_size}, block_size={block_size}")
    # Initialize flag
    flag = np.array([0], dtype=np.int32)
    d_flag = cuda.to_device(flag)
    try:
        # Launch kernel with flag
        simulate_games_kernel[grid_size, block_size](d_board_states, d_current_players, d_results, num_simulations, d_seeds, d_flag)
        # Wait for the kernel to finish
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
    # Copy flag back to host
    #flag = d_flag.copy_to_host()
    #if flag[0] == 1:
        #print("Kernel executed at least one thread.")
    #else:
        #print("Kernel did not execute any thread.")
    # Copy results back to host
    results = d_results.copy_to_host()
    #print(f"Simulation completed. Results sample: {results[:10]}")  # Print first 10 results for verification
    return results

# ----------------------------- MCTS Classes ----------------------------- #
class MCTS:
    def __init__(self, num_simulations=4096, debug=False):
        self.num_simulations = num_simulations
        self.debug = debug

    def check_immediate_win(self, env, player):
        """Return the column that immediately wins for 'player', or None if none."""
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == player:
                return col
        return None

    def check_immediate_block(self, env, player):
        """Return the column that blocks an immediate win by the opponent, or None if none."""
        opponent = 3 - player
        valid_actions = env.get_valid_actions()
        for col in valid_actions:
            temp_env = env.copy()
            temp_env.make_move(col)
            if temp_env.check_winner() == opponent:
                return col
        return None

    def select_action(self, env, current_player):
        # 1) Check forced win
        move = self.check_immediate_win(env, current_player)
        if move is not None:
            if self.debug:
                logging.info(f"Found immediate win by playing column {move}")
            return move

        # 2) Check forced block
        move = self.check_immediate_block(env, current_player)
        if move is not None:
            if self.debug:
                logging.info(f"Found immediate block by playing column {move}")
            return move

        if self.debug:
            logging.info(f"Running MCTS with {self.num_simulations} simulations using CUDA.")

        # 3) Run simulations on GPU
        simulation_results = run_simulations_cuda(env, self.num_simulations)
        if simulation_results is None:
            if self.debug:
                logging.error("Simulations failed. Returning random action.")
            # Fallback to random action
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                return None
            return random.choice(valid_actions)

        # 4) Aggregate results per action
        # Distribute simulations equally among actions
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None  # No valid moves

        simulations_per_action = self.num_simulations // len(valid_actions)
        extra_simulations = self.num_simulations % len(valid_actions)

        action_results = {}
        for action in valid_actions:
            # Make the move
            temp_env = env.copy()
            temp_env.make_move(action)
            # Determine the number of simulations for this action
            sims = simulations_per_action + (1 if extra_simulations > 0 else 0)
            if extra_simulations > 0:
                extra_simulations -=1
            if sims == 0:
                continue  # Skip if no simulations allocated
            # Run simulations from this state
            simulation_outcomes = run_simulations_cuda(temp_env, sims)
            if simulation_outcomes is None:
                if self.debug:
                    logging.warning(f"Simulations for action {action} failed. Skipping.")
                continue
            # Count wins, draws, and losses
            wins = np.sum(simulation_outcomes == PLAYER1) if current_player == PLAYER1 else np.sum(simulation_outcomes == PLAYER2)
            draws = np.sum(simulation_outcomes == EMPTY)
            losses = sims - wins - draws
            # Aggregate results: assign full point for wins, half point for draws
            action_results[action] = wins + 0.5 * draws

        if self.debug:
            logging.debug(f"Aggregated action results: {action_results}")

        if not action_results:
            if self.debug:
                logging.warning("No valid action results after simulations. Choosing random action.")
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                return None
            return random.choice(valid_actions)

        # Choose the action with the highest score
        best_action = max(action_results, key=action_results.get)

        if self.debug:
            logging.debug(f"Chose best action {best_action}")

        return best_action
