import random
import math
import logging
import multiprocessing as mp
from dependencies.environment import Connect4

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------- MCTS Classes ---------------------- #
class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state.copy()  # Instance of Connect4
        self.parent = parent
        self.action_taken = action_taken
        self.visits = 0
        self.wins = 0
        self.children = []
        # The player who made the move to reach this node
        self.player = 3 - state.current_player  # Assuming current_player is the one to make the next move

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

def run_mcts(env, num_simulations=10, debug=False):
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
                new_state = node.state.copy()
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
        sim_state = node.state.copy()

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
    def __init__(self, num_simulations=10, debug=0, num_processes=4):
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
    args = [(env.copy(), simulations_per_process, debug) for _ in range(num_processes)]
    if remaining_simulations > 0:
        args.append((env.copy(), remaining_simulations, debug))
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

def simulate_games(num_games, sims_agent1, sims_agent2, debug=False):
    """
    Simulate games between two MCTS agents with different numbers of simulations.
    """
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0

    for game in range(num_games):
        env = Connect4()
        current_player = 1
        agents = {
            1: Parallel_MCTS(num_simulations=sims_agent1, debug=debug),
            2: Parallel_MCTS(num_simulations=sims_agent2, debug=debug)
        }

        while True:
            action = agents[current_player].select_action(env)
            if action is None:
                draws += 1
                if debug:
                    logging.debug("No valid actions available. Game is a draw.")
                break

            env.make_move(action)
            winner = env.check_winner()
            if winner:
                if winner == 1:
                    wins_agent1 += 1
                    if debug:
                        logging.debug("Agent 1 wins the game.")
                else:
                    wins_agent2 += 1
                    if debug:
                        logging.debug("Agent 2 wins the game.")
                break
            if env.is_draw():
                draws += 1
                if debug:
                    logging.debug("Game ended in a draw.")
                break

            current_player = 3 - current_player

        if (game + 1) % 5 == 0 and debug:
            logging.info(f"Game {game + 1}/{num_games} completed")

    return {
        "wins_agent1": wins_agent1,
        "wins_agent2": wins_agent2,
        "draws": draws
    }


# ---------------------- main ---------------------- #
def main():
    # Main Simulation
    simulation_counts = [1000, 1250,1500,1750, 2000,2100,2200,2300,2400, 2500]
    num_games_per_sim = 100
    sims_agent2 = 800
    debug_mode = False  # Set to True to enable detailed logging

    for sims_agent1 in simulation_counts:
        logging.info(f"Testing Agent 1 with {sims_agent1} simulations against Agent 2 with {sims_agent2} simulations")
        results = simulate_games(num_games_per_sim, sims_agent1, sims_agent2, debug=debug_mode)
        win_rate_agent1 = results["wins_agent1"] / num_games_per_sim
        win_rate_agent2 = results["wins_agent2"] / num_games_per_sim

        logging.info(f"Agent 1 (sims={sims_agent1}) Win Rate: {win_rate_agent1:.2f}")
        logging.info(f"Agent 2 (sims={sims_agent2}) Win Rate: {win_rate_agent2:.2f}")
        logging.info(f"Draw Rate: {results['draws'] / num_games_per_sim:.2f}")
        if results["wins_agent2"] >  50:
            logging.info(f"Optimized Value of the Simulation of MCTS:{sims_agent2}")
            break
        if results["wins_agent2"] < num_games_per_sim - 5:
            logging.info(f"Updating Agent 2's simulations to {sims_agent1}")
            sims_agent2 = sims_agent1
        

if __name__ == "__main__":
    main()
