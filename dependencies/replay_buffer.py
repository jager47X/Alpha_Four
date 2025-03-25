import os
import numpy as np
import torch
from dependencies.utils import safe_make_dir

class DiskReplayBuffer:
    def __init__(self, capacity, state_shape=(6, 7), 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 version=0):
        """
        Initialize a disk-based replay buffer using memory-mapped files.
        Transition order:
            state, action, reward, next_state, done flag, best_q_val,
            mcts_value, hybrid_value, mcts_action, model_used
        Stored in `data/dat/{version}`.
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.ptr = 0
        self.full = False

        # Define directory for replay buffer files.
        self.data_dir = os.path.join("data", "dat", str(version))
        safe_make_dir(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        def init_memmap(filename, dtype, shape, mode="w+"):
            file_path = os.path.join(self.data_dir, filename)
            mode = "r+" if os.path.exists(file_path) else mode
            return np.memmap(file_path, dtype=dtype, mode=mode, shape=shape)

        # Allocate memory-mapped arrays for each field.
        self.states = init_memmap("replay_buffer_states.dat", np.float32, (capacity, *state_shape))
        self.actions = init_memmap("replay_buffer_actions.dat", np.int32, (capacity,))
        self.rewards = init_memmap("replay_buffer_rewards.dat", np.float32, (capacity,))  # reward
        self.next_states = init_memmap("replay_buffer_next_states.dat", np.float32, (capacity, *state_shape))
        self.dones = init_memmap("replay_buffer_dones.dat", bool, (capacity,))
        self.best_q_vals = init_memmap("replay_buffer_best_q_vals.dat", np.float32, (capacity,))
        self.mcts_values = init_memmap("replay_buffer_mcts_values.dat", np.float32, (capacity,))
        self.hybrid_values = init_memmap("replay_buffer_hybrid_values.dat", np.float32, (capacity,))
        self.mcts_actions = init_memmap("replay_buffer_mcts_actions.dat", np.int32, (capacity,))
        # Store model_used as an integer code: 0: "dqn", 1: "mcts", 2: "hybrid", 3: None
        self.model_used = init_memmap("replay_buffer_model_used.dat", np.int32, (capacity,))

    def default_if_none(self, value, default=-1):
        """Return value if it is not None; otherwise, return the default."""
        return value if value is not None else default

    def push(self, state, action, reward, next_state, done, 
             best_q_val, mcts_value, hybrid_value, mcts_action, model_used):
        """
        Add a transition to the replay buffer.
        Expected order:
            state, action, reward, next_state, done, best_q_val,
            mcts_value, hybrid_value, mcts_action, model_used
        model_used should be a string ("dqn", "mcts", "hybrid") or None.
        """
        # Apply default value (-1) for any None value
        state_val         = self.default_if_none(state)
        action_val        = self.default_if_none(action)
        reward_val        = self.default_if_none(reward)
        next_state_val    = self.default_if_none(next_state)
        done_val          = self.default_if_none(done)
        best_q_val_val    = self.default_if_none(best_q_val)
        mcts_value_val    = self.default_if_none(mcts_value)
        hybrid_value_val  = self.default_if_none(hybrid_value)
        mcts_action_val   = self.default_if_none(mcts_action)

        self.states[self.ptr]       = state_val
        self.actions[self.ptr]        = action_val
        self.rewards[self.ptr]        = reward_val
        self.next_states[self.ptr]    = next_state_val
        self.dones[self.ptr]          = done_val
        self.best_q_vals[self.ptr]    = best_q_val_val
        self.mcts_values[self.ptr]    = mcts_value_val
        self.hybrid_values[self.ptr]  = hybrid_value_val
        self.mcts_actions[self.ptr]   = mcts_action_val

        # Convert model_used string to integer code.
        if model_used is None:
            code = 3
        elif model_used == "dqn":
            code = 0
        elif model_used == "mcts":
            code = 1
        elif model_used == "hybrid":
            code = 2
        else:
            code = 3
        self.model_used[self.ptr] = code

        # Flush changes to disk.
        self.states.flush()
        self.actions.flush()
        self.rewards.flush()
        self.next_states.flush()
        self.dones.flush()
        self.best_q_vals.flush()
        self.mcts_values.flush()
        self.hybrid_values.flush()
        self.mcts_actions.flush()
        self.model_used.flush()

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
            "q_actions": torch.tensor(self.actions[indices], device=self.device, dtype=torch.int64),
            "rewards": torch.tensor(self.rewards[indices], device=self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[indices], device=self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices], device=self.device, dtype=torch.bool),
            "best_q_vals": torch.tensor(self.best_q_vals[indices], device=self.device, dtype=torch.float32),
            "mcts_values": torch.tensor(self.mcts_values[indices], device=self.device, dtype=torch.float32),
            "hybrid_values": torch.tensor(self.hybrid_values[indices], device=self.device, dtype=torch.float32),
            "mcts_actions": torch.tensor(self.mcts_actions[indices], device=self.device, dtype=torch.int64),
            "model_used": torch.tensor(self.model_used[indices], device=self.device, dtype=torch.int64),
        }
        return batch

    def __len__(self):
        return self.capacity if self.full else self.ptr
