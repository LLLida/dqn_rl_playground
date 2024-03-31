import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DinoAgent(nn.Module):

    def __init__(self, state_dim: int, epsilon=0, nactions: int = 2):
        super().__init__()

        self.state_dim = state_dim
        self.nactions = nactions
        self.epsilon = epsilon

        self.dense1 = nn.Linear(state_dim, 32)
        self.dense2 = nn.Linear(32, 32)
        self.dense3 = nn.Linear(32, nactions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        assert(state.shape[-1] == self.state_dim)

        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        qvalues = self.dense3(x)

        return qvalues

    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        """
        like forward, but works on numpy arrays
        """
        device = next(self.parameters()).device
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """
        pick actions given qvalues. Uses epsilon-greedy exploration strategy.
        """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice([0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
