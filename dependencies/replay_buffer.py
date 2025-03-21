import os
import numpy as np
import torch

class DiskReplayBuffer:
    def __init__(self, capacity, state_shape=(6, 7), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize a disk-based replay buffer using memory-mapped files.
        Stores replay buffer files in `data/dat/` following the specified project structure.
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.ptr = 0
        self.full = False

        # Define the directory where replay buffer `.dat` files will be stored
        self.data_dir = os.path.join("data", "dat")
        os.makedirs(self.data_dir, exist_ok=True)  # Ensure the directory exists

        def init_memmap(filename, dtype, shape, mode="w+"):
            file_path = os.path.join(self.data_dir, filename)  # Store inside `data/dat/`
            mode = "r+" if os.path.exists(file_path) else mode
            return np.memmap(file_path, dtype=dtype, mode=mode, shape=shape)

        # Save all replay buffer data in `data/dat/`
        self.states = init_memmap("replay_buffer_states.dat", np.float32, (capacity, *state_shape))
        self.actions = init_memmap("replay_buffer_actions.dat", np.int32, (capacity,))
        self.rewards = init_memmap("replay_buffer_rewards.dat", np.float32, (capacity,))
        self.next_states = init_memmap("replay_buffer_next_states.dat", np.float32, (capacity, *state_shape))
        self.dones = init_memmap("replay_buffer_dones.dat", bool, (capacity,))
        self.mcts_values = init_memmap("replay_buffer_mcts_values.dat", np.float32, (capacity,))

    def push(self, state, action, reward, next_state, done, mcts_value):
        """
        Add a transition along with its MCTS value to the replay buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.mcts_values[self.ptr] = mcts_value

        # Flush to disk
        self.states.flush()
        self.actions.flush()
        self.rewards.flush()
        self.next_states.flush()
        self.dones.flush()
        self.mcts_values.flush()

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size):
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
            "mcts_values": torch.tensor(self.mcts_values[indices], device=self.device, dtype=torch.float32),
        }
        return batch

    def __len__(self):
        return self.capacity if self.full else self.ptr
