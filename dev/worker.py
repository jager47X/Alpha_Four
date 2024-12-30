import random
import numpy as np
from connect4 import Connect4
from AgentLogic import AgentLogic

def run_single_episode(policy_net_1, policy_net_2, replay_buffer_1, replay_buffer_2, EPSILON,current_episode):
    """
    Runs a single Connect4 episode and returns the results.

    Args:
        policy_net_1: Policy network for Agent 1.
        policy_net_2: Policy network for Agent 2.
        replay_buffer_1: Replay buffer for Agent 1.
        replay_buffer_2: Replay buffer for Agent 2.
        EPSILON: Epsilon value for epsilon-greedy exploration.

    Returns:
        total_reward_1: Total reward for Agent 1.
        total_reward_2: Total reward for Agent 2.
        winner: The winner of the game (1, 2, or 0 for a draw).
    """
    env = Connect4()
    state = env.reset()
    done = False
    total_reward_1, total_reward_2 = 0, 0

    agent_logic_1 = AgentLogic(policy_net_1)
    agent_logic_2 = AgentLogic(policy_net_2)

    while not done:
        action_1 = agent_logic_1.combined_action(env,current_episode)
        if EPSILON > random.random():
            action_1 = random.choice(env.get_valid_actions())
        env.make_move(action_1, warning=1)
        reward_1, win_status = agent_logic_1.calculate_reward(env, action_1, current_player=1)

        next_state = env.board.copy()
        replay_buffer_1.append((state, action_1, reward_1, next_state, done))
        state = next_state
        total_reward_1 += reward_1
        done = win_status != 0 or env.is_draw()

        if not done:
            action_2 = agent_logic_2.combined_action(env,current_episode)
            if EPSILON > random.random():
                action_2 = random.choice(env.get_valid_actions())
            env.make_move(action_2, warning=1)
            reward_2, win_status = agent_logic_2.calculate_reward(env, action_2, current_player=2)

            next_state = env.board.copy()
            replay_buffer_2.append((state, action_2, reward_2, next_state, done))
            state = next_state
            total_reward_2 += reward_2
            done = win_status != 0 or env.is_draw()

    return total_reward_1, total_reward_2, env.check_winner()
