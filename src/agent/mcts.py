# mcts_agent.py

import math
import multiprocessing as mp
import logging
import random
from copy import deepcopy
from env.connect4 import Connect4  # Ensure the Connect4 environment is accessible

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = deepcopy(state)  # Deep copy to avoid mutation
        self.parent = parent
        self.action_taken = action_taken
        self.visits = 0
        self.wins = 0
        self.children = []
        # The player who made the move to reach this node
        self.player = 3 - self.state.current_player  # Assuming current_player is the one to make the next move

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

def run_mcts(env, num_simulations=100, debug=False):
    """
    Enhanced MCTS implementation with proper backpropagation.
    """
    root = MCTSNode(env)

    for sim in range(1, num_simulations + 1):
        node = root

        # 1) Selection
        while node.children:
            node = max(node.children, key=lambda c: c.ucb())
            if debug:
                logging.debug(f"Selected node with action {node.action_taken} by player {node.player}")

        # 2) Expansion
        winner_now = node.state.check_winner()
        if winner_now == 0 and not node.state.is_draw():
            valid_actions = node.state.get_valid_actions()
            for act in valid_actions:
                new_state = deepcopy(node.state)
                new_state.make_move(act)
                child = MCTSNode(new_state, parent=node, action_taken=act)
                node.children.append(child)
                if debug:
                    logging.debug(f"Expanded node with action {act} by player {child.player}")

            # Pick one child at random to rollout
            if node.children:
                node = random.choice(node.children)
                winner_now = node.state.check_winner()
                if debug:
                    logging.debug(f"Chose to rollout child with action {node.action_taken}")

        # 3) Simulation (random playout)
        sim_state = deepcopy(node.state)

        while sim_state.check_winner() == 0 and not sim_state.is_draw():
            vacts = sim_state.get_valid_actions()
            if not vacts:
                break
            choice = random.choice(vacts)
            sim_state.make_move(choice)

        final_winner = sim_state.check_winner()
        if debug:
            logging.debug(f"Simulation ended with winner: {final_winner}")

        # 4) Backpropagation
        current = node
        while current is not None:
            current.visits += 1
            if final_winner == current.player:
                current.wins += 1
            elif final_winner == 0:
                current.wins += 0.5  # Assign half-win for draws
            else:
                current.wins -= 1
            if debug:
                logging.debug(f"Backpropagated to player {current.player} - Visits: {current.visits}, Wins: {current.wins}")
            current = current.parent

        if debug and sim % 10 == 0:
            logging.info(f"Completed {sim}/{num_simulations} simulations")

    # Pick child with max visits
    if not root.children:
        # No expansion => terminal or no valid moves
        valid_actions = env.get_valid_actions()
        if valid_actions:
            chosen_action = random.choice(valid_actions)
            if debug:
                logging.debug(f"No children in root. Chose random action {chosen_action}")
            return chosen_action
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    if debug:
        logging.debug(f"Chose best action {best_child.action_taken} with {best_child.visits} visits")
    return best_child.action_taken

class Parallel_MCTS:
    def __init__(self, num_simulations=100, debug=False, num_processes=4):
        self.num_simulations = num_simulations
        self.debug = debug
        self.num_processes = num_processes

    def select_action(self, env):
        if self.debug:
            logging.info(f"Running MCTS with {self.num_simulations} simulations using {self.num_processes} processes.")
        return run_parallel_mcts(env, self.num_simulations, self.num_processes, self.debug)

# ---------------------- Multiprocessing Setup ---------------------- #
def parallel_mcts_search(args):
    env, num_simulations, debug = args
    return run_mcts(env, num_simulations, debug)

def run_parallel_mcts(env, total_simulations, num_processes, debug=False):
    simulations_per_process = total_simulations // num_processes
    remaining_simulations = total_simulations % num_processes
    args = [(deepcopy(env), simulations_per_process, debug) for _ in range(num_processes)]
    if remaining_simulations > 0:
        args.append((deepcopy(env), remaining_simulations, debug))
    try:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(parallel_mcts_search, args)
    except Exception as e:
        logging.error(f"Multiprocessing pool encountered an error: {e}")
        return None

    # Aggregate results
    action_counts = {}
    for action in results:
        if action is not None:
            action_counts[action] = action_counts.get(action, 0) + 1

    # Choose the action with the highest count
    if not action_counts:
        return random.choice(env.get_valid_actions()) if env.get_valid_actions() else None
    best_action = max(action_counts, key=action_counts.get)
    if debug:
        logging.debug(f"Aggregated action counts: {action_counts}")
        logging.debug(f"Chose best action {best_action}")
    return best_action

class MCTS:
    def __init__(self, num_simulations=100, num_processes=4, debug=False):
        self.mcts = Parallel_MCTS(num_simulations=num_simulations, debug=debug, num_processes=num_processes)

    def pick_action(self, env):
        return self.mcts.select_action(env)
