from concurrent.futures import ProcessPoolExecutor
from config import (
    BATCH_SIZE, GAMMA, LEARNING_RATE, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    REPLAY_BUFFER_SIZE, TARGET_UPDATE, NUM_EPISODES, TRAINER_SAVE_PATH, MODEL_SAVE_PATH, device
)
from TrainAgent import setup_logger, train_agent
from connect4 import Connect4
from AgentLogic import AgentLogic
from utils import load_model_checkpoint, save_model
from worker import run_single_episode
import os

# Number of workers for ProcessPoolExecutor
num_workers = os.cpu_count()

# Update target networks, decay epsilon, and save models periodically
def periodic_updates(
        episode, policy_net_1, target_net_1, policy_net_2, target_net_2,
        optimizer_1, optimizer_2, TRAINER_SAVE_PATH, MODEL_SAVE_PATH, EPSILON, 
        EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger):
    try:
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())
            logger.info("Target networks updated")

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        logger.info(f"Epsilon decayed to {EPSILON}")

        if episode % TARGET_UPDATE == 0:
            save_model(TRAINER_SAVE_PATH, policy_net_1, optimizer_1, episode, logger)
            save_model(MODEL_SAVE_PATH, policy_net_2, optimizer_2, episode, logger)
            logger.info(f"Models saved at episode {episode}")

    except Exception as e:
        logger.error(f"Error during periodic updates at episode {episode}: {e}")
    return EPSILON

def main():
    logger = setup_logger("log.txt")
    
    policy_net_1, target_net_1, optimizer_1, replay_buffer_1, start_episode_1 = load_model_checkpoint(
        TRAINER_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )
    policy_net_2, target_net_2, optimizer_2, replay_buffer_2, start_episode_2 = load_model_checkpoint(
        MODEL_SAVE_PATH, None, None, None, None, LEARNING_RATE, REPLAY_BUFFER_SIZE, logger, device
    )


    global EPSILON

    agent_1_wins, agent_2_wins, draws = 0, 0, 0
    GAMES_PER_BATCH = 100

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for episode in range(1, NUM_EPISODES + 1, GAMES_PER_BATCH):
            futures = [
                executor.submit(
                    run_single_episode,
                    policy_net_1,
                    policy_net_2,
                    replay_buffer_1,
                    replay_buffer_2,
                    EPSILON,
                    episode
                )
                for _ in range(GAMES_PER_BATCH)
            ]

            for future in futures:
                total_reward_1, total_reward_2, winner = future.result()

                if winner == 1:
                    agent_1_wins += 1
                elif winner == 2:
                    agent_2_wins += 1
                else:
                    draws += 1

            logger.info(
                f"Batch {episode}-{episode + GAMES_PER_BATCH - 1}: Trainer Wins: {agent_1_wins}, "
                f"Agent Wins: {agent_2_wins}, Draws: {draws}, EPSILON: {EPSILON}"
                f"Trainer total reward: {total_reward_1}, Agent total reward:{total_reward_2} "
            )

            if len(replay_buffer_1) >= BATCH_SIZE:
                train_agent(policy_net_1, target_net_1, optimizer_1, replay_buffer_1)
            if len(replay_buffer_2) >= BATCH_SIZE:
                train_agent(policy_net_2, target_net_2, optimizer_2, replay_buffer_2)

            EPSILON = periodic_updates(
                episode, policy_net_1, target_net_1, policy_net_2, target_net_2,
                optimizer_1, optimizer_2, TRAINER_SAVE_PATH, MODEL_SAVE_PATH,
                EPSILON, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE, logger
            )

    logger.info(f"Training complete. Agent 1 Wins: {agent_1_wins}, Agent 2 Wins: {agent_2_wins}, Draws: {draws}")

if __name__ == "__main__":
    main()
