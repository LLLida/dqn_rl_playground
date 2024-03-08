import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TetrisAgent(nn.Module):

    def __init__(self, nrows: int, ncols: int, nactions: int, epsilon=0):
        super().__init__()

        self.nrows = ncols
        self.ncols = ncols
        self.nactions = nactions
        self.epsilon = epsilon


        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)

        w, h = ncols, nrows
        w = conv2d_size_out(w, kernel_size=3, stride=1); h = conv2d_size_out(h, kernel_size=3, stride=2)
        w = conv2d_size_out(w, kernel_size=3, stride=1); h = conv2d_size_out(h, kernel_size=3, stride=1)

        self.dense1 = nn.Linear(w*h*16, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, nactions)

    def forward(self, state_t: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state_t))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        qvalues = F.relu(self.dense3(x))

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert (
            len(qvalues.shape) == 2 and
            qvalues.shape[0] == state_t.shape[0] and
            qvalues.shape[1] == self.nactions
        )

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

def conv2d_size_out(size, kernel_size, stride):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1
