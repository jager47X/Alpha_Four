import numpy as np
import torch

class DiskReplayBuffer:
    """
    A Replay Buffer that stores data on disk via NumPy memmap files.
    This helps avoid running out of CPU/GPU RAM for large buffers.
    """
    def __init__(
        self,
        capacity: int,
        state_shape=(6, 7),
        prefix_path="./replay_buffer",
        device="cpu"
    ):
        """
        Args:
            capacity (int): Max number of transitions to store.
            state_shape (tuple): Shape of each state (6,7) for Connect4.
            prefix_path (str): Prefix for the .dat files on disk.
            device (str): 'cpu' or 'cuda'.
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.ptr = 0
        self.full = False
        self.prefix_path=prefix_path

        # Create memmap files for states, next_states, actions, rewards, dones
        self.states = np.memmap(
            f"{self.prefix_path}_states.dat",
            dtype=np.float32,
            mode="w+",
            shape=(capacity, *state_shape),
        )
        self.actions = np.memmap(
            f"{self.prefix_path}_actions.dat",
            dtype=np.int32,
            mode="w+",
            shape=(capacity,),
        )
        self.rewards = np.memmap(
            f"{self.prefix_path}_rewards.dat",
            dtype=np.float32,
            mode="w+",
            shape=(capacity,),
        )
        self.next_states = np.memmap(
            f"{self.prefix_path}_next_states.dat",
            dtype=np.float32,
            mode="w+",
            shape=(capacity, *state_shape),
        )
        self.dones = np.memmap(
            f"{self.prefix_path}_dones.dat",
            dtype=np.bool_,
            mode="w+",
            shape=(capacity,),
        )

    def push(self, state, action, reward, next_state, done):
        """Store one transition. Overwrites oldest if at capacity."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Advance pointer
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size):
        """Sample a random batch from the buffer, returning torch Tensors."""
        max_idx = self.capacity if self.full else self.ptr
        if batch_size > max_idx:
            raise ValueError(f"Not enough samples: have {max_idx}, need {batch_size}.")

        idxs = np.random.choice(max_idx, batch_size, replace=False)

        states_batch = self.states[idxs]
        actions_batch = self.actions[idxs]
        rewards_batch = self.rewards[idxs]
        next_states_batch = self.next_states[idxs]
        dones_batch = self.dones[idxs]

        states_tensor = torch.tensor(states_batch, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states_batch, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.bool, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
