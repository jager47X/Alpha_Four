from concurrent.futures import ProcessPoolExecutor
import torch
from config import (
    BATCH_SIZE, GAMMA, LEARNING_RATE, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    REPLAY_BUFFER_SIZE, TARGET_UPDATE, NUM_EPISODES, TRAINER_SAVE_PATH, MODEL_SAVE_PATH, device
)
from TrainAgent import setup_logger, train_agent
from connect4 import Connect4
from AgentLogic import AgentLogic
from logger_utils import setup_logger
from DQN import DQN
from utills import  load_model_checkpoint, save_model
import os
num_workers = os.cpu_count()

# Update target networks, decay epsilon, and save models periodically
def periodic_updates(
        episode, policy_net_1, target_net_1, policy_net_2, target_net_2,
                    optimizer_1,optimizer_2,
                     TRAINER_SAVE_PATH, MODEL_SAVE_PATH, EPSILON, EPSILON_MIN, 
                     EPSILON_DECAY, TARGET_UPDATE, logger):
    try:
        # Update target networks periodically
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        # Save models periodically
        if episode % TARGET_UPDATE == 0:
            save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
            save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
            logger.info(f"Models saved at episode {episode}")

    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON



def run_episode(env, agent_logic_1, agent_logic_2):
    """
    Run a single Connect4 game between two agents using AgentLogic and calculate rewards.
    """
    current_player = 1
    total_rewards = {1: 0.0, 2: 0.0}
    done = False

    while not done:
        # Determine which agent to use
        agent_logic = agent_logic_1 if current_player == 1 else agent_logic_2

        # Get action from the agent
        action = agent_logic.combined_action(env)

        # Make the move
        env.make_move(action, warning=1)

        # Calculate reward using AgentLogic's calculate_reward
        reward, winner = agent_logic.calculate_reward(env, action, current_player)
        total_rewards[current_player] += reward

        if winner != 0 or env.is_draw():
            done = True  # End the game if there's a winner or a draw
        else:
            current_player = 3 - current_player  # Switch player

    return winner, total_rewards

def main():
    # Load or initialize models
    policy_net_1, target_net_1, optimizer_1, replay_buffer_1, start_episode_1 = load_model_checkpoint(
        TRAINER_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )
    policy_net_2, target_net_2, optimizer_2, replay_buffer_2, start_episode_2 = load_model_checkpoint(
        MODEL_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )

    # Initialize AgentLogic for both agents
    agent_1_logic = AgentLogic(policy_net_1)
    agent_2_logic = AgentLogic(policy_net_2)

    global EPSILON
    env = Connect4()
    agent_1_wins, agent_2_wins, draws = 0, 0, 0

    # Number of concurrent games
    GAMES_PER_BATCH = 100

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for episode in range(1, NUM_EPISODES + 1, GAMES_PER_BATCH):
            # Submit 100 games concurrently
            futures = [
                executor.submit(run_episode, env.copy(), agent_1_logic, agent_2_logic)
                for _ in range(GAMES_PER_BATCH)
            ]

            # Collect results as they complete
            for future in futures:
                winner, total_rewards = future.result()

                # Update win/draw statistics
                if winner == 1:
                    agent_1_wins += 1
                elif winner == 2:
                    agent_2_wins += 1
                else:
                    draws += 1

            logger.info(
                f"Batch {episode}-{episode + GAMES_PER_BATCH - 1}: Agent 1 Wins: {agent_1_wins}, "
                f"Agent 2 Wins: {agent_2_wins}, Draws: {draws}, EPSILON: {EPSILON}, Total_Reward_1:{total_rewards[1]}, Total_Reward_2:{total_rewards[2]}"
            )

            # Train agents after each batch
            if len(replay_buffer_1) >= BATCH_SIZE:
                train_agent(policy_net_1, target_net_1, optimizer_1, replay_buffer_1)
            if len(replay_buffer_2) >= BATCH_SIZE:
                train_agent(policy_net_2, target_net_2, optimizer_2, replay_buffer_2)

            # Perform periodic updates after each batch
            EPSILON = periodic_updates(
                episode, policy_net_1, target_net_1, policy_net_2, target_net_2,
                optimizer_1, optimizer_2, TRAINER_SAVE_PATH, MODEL_SAVE_PATH,
                EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
            )

    logger.info(f"Training complete. Agent 1 Wins: {agent_1_wins}, Agent 2 Wins: {agent_2_wins}, Draws: {draws}")


if __name__ == "__main__":
    # Initialize logging
    logger = setup_logger("log.txt")
    main()
