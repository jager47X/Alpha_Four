# Connect4 DQN Self-Learning Project

## Overview
The **Connect4 DQN Self-Learning Project** trains two agents, Agent1 and Agent2, to play Connect4 through competitive self-play. The project focuses on the **Deep Q-Network (DQN)** model for reinforcement learning and integrates a logic-based agent to guide initial gameplay and improve learning.

## Features

### Agent1 (Logic-Based Agent)
- Makes moves based on predefined rules:
  - Identifies **winning moves**.
  - Blocks **opponent's winning moves**.
  - Utilizes **Monte Carlo Tree Search (MCTS)** for simulations when no logical moves are available.
- Acts as a strategic baseline for Agent2.

### Agent2 (DQN-Based Agent)
- Learns optimal strategies using a **Deep Q-Network**:
  - Predicts Q-values for actions based on the current board state.
  - Combines **exploration** (random actions) and **exploitation** (Q-value predictions) to improve gameplay.
  - Augments decision-making with logic-based actions.
  
### Replay Buffer
- A **disk-based replay buffer** efficiently stores gameplay experiences:
  - Experience format: `(state, action, reward, next_state, done)`.
  - Scales to large data sizes using memmap, avoiding memory limitations.

### Training Dynamics
- **Turn-Based Gameplay**: Alternates moves between Agent1 and Agent2.
- **Training**:
  - Agent1 uses logic and MCTS strategies.
  - Agent2 learns from Q-values and random exploration using reinforcement learning.
- **Target Network Updates**: Stabilizes training with periodic synchronization between the policy and target networks.

## Deep Q-Learning Model (DQN)

### Architecture
- **Input Layer**: Takes a flattened representation of the 6x7 game board.
- **Hidden Layers**:
  - Fully connected layers with ReLU activations.
  - Dropout for regularization.
  - Batch normalization for stabilizing gradients.
- **Output Layer**: Predicts Q-values for 7 possible actions (columns).

### Training Workflow
1. **Batch Sampling**: Experiences are sampled randomly from the replay buffer.
2. **Bellman Equation**:
   \[
   Q_{\text{target}} = r + \gamma \max_a Q(s', a)
   \]
   Calculates target Q-values using the next state's maximum Q-value.
3. **Loss Function**: Mean Squared Error (MSE) between predicted and target Q-values.
4. **Backpropagation**: Updates network weights using an optimizer.

## Self-Learning Workflow

1. **Initialization**:
   - Two agents (Agent1 and Agent2) are initialized.
   - Replay buffers and neural networks are set up.
   
2. **Gameplay**:
   - Agents alternate turns:
     - **Agent1**: Logic-based decisions.
     - **Agent2**: Combines Q-values and random exploration (epsilon-greedy).
   - Experiences are stored in replay buffers.

3. **Training**:
   - After each episode, both agents train on replay buffer batches.
   - Target networks are updated periodically.

4. **Evaluation**:
   - Win rates and rewards for both agents are logged and visualized.

## Visualization of Learning Process of the Model

### Win Rate Plot
- Tracks win rates and draws for Agent1 and Agent2 over episodes.
- Interval-based annotations highlight performance improvements.

### Dynamic Annotations
- Calculates and displays win rates for Agent1 and Agent2 at regular intervals (e.g., every 100 episodes).
- Visualizes the learning progress of Agent2.

## Key Enhancements
- **Disk-Based Replay Buffer**: Efficient management of training data.
- **Logic-Based Initialization**: Combines logic-based gameplay with Q-learning for a smoother learning curve.
- **Monte Carlo Tree Search (MCTS)**: Enhances decision-making during early training stages.

## Usage

1. **Run Training**:
   - Execute the main script to train both agents:
     ```bash
     python cfrl_self_learning.py
     ```

2. **Visualize Results**:
   - Use the plotting script to analyze performance metrics:
     ```bash
     python plot_Process.py
     ```
## Result
- **Perfomance against human**:
    - Random: WinRate <0.001%
    - MCTS: WinRate 50%
    - CFSL_Model 7: WinRate 70%
## Future Improvements
- **Advanced Architectures**:
  - Incorporate convolutional layers to better capture spatial patterns in the game board.
  - Explore Double DQN and Dueling DQN architectures for improved Q-value estimation.
  - Add more detailed reward calculation mechanism to enhance the learning process.
- **Enhanced Visualization**:
  - Add real-time plotting to monitor agent performance during training.
- **Priority Replay**:
  - Implement prioritized experience replay to improve learning efficiency.

---

This project demonstrates the integration of logic-based decision-making and deep reinforcement learning in a competitive self-learning framework for Connect4. The DQN agent (Agent2) is the central focus, while Agent1 provides a logical baseline for improvement.
