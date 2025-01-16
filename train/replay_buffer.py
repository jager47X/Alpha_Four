import os
import numpy as np
import torch

class DiskReplayBuffer:
    def __init__(self, capacity, state_shape=(6, 7), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize a disk-based replay buffer using memory-mapped files.

        Args:
            capacity (int): Maximum number of transitions the buffer can store.
            state_shape (tuple): Shape of the state (default: (6, 7) for a Connect 4 board).
            device (str): Device to which data will be transferred (e.g., 'cpu' or 'cuda').
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.ptr = 0
        self.full = False

        def init_memmap(filename, dtype, shape, mode="w+"):
            """Initialize or load a memory-mapped file."""
            mode = "r+" if os.path.exists(filename) else mode
            return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

        # Initialize memory-mapped files for replay buffer data.
        self.states = init_memmap("replay_buffer_states.dat", np.float32, (capacity, *state_shape))
        self.actions = init_memmap("replay_buffer_actions.dat", np.int32, (capacity,))
        self.rewards = init_memmap("replay_buffer_rewards.dat", np.float32, (capacity,))
        self.next_states = init_memmap("replay_buffer_next_states.dat", np.float32, (capacity, *state_shape))
        self.dones = init_memmap("replay_buffer_dones.dat", bool, (capacity,))

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer and flush changes to disk.

        If the buffer is full, it overwrites the oldest data.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is done.
        """
        # Store the transition at the current pointer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Flush changes to disk to ensure data integrity
        self.states.flush()
        self.actions.flush()
        self.rewards.flush()
        self.next_states.flush()
        self.dones.flush()

        # Update the pointer and handle overwriting if necessary
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: Batch of sampled transitions.
        """
        max_index = self.capacity if self.full else self.ptr
        if max_index == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        replace = False if self.full else True if batch_size > max_index else False
        indices = np.random.choice(max_index, batch_size, replace=replace)

        batch = {
            "states": torch.tensor(self.states[indices], device=self.device, dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices], device=self.device, dtype=torch.int64),
            "rewards": torch.tensor(self.rewards[indices], device=self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[indices], device=self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices], device=self.device, dtype=torch.bool),
        }
        return batch

    def __len__(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            int: Number of elements in the buffer.
        """
        return self.capacity if self.full else self.ptr