import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import fftconvolve, gaussian
import torch
from tqdm import trange

from tetris.game import TetrisGame
from tetris.agent import TetrisAgent
import rl.training as T
from rl.replay_buffer import ReplayBuffer

agent_save_path = 'tetris.pkl'
exp_replay_save_path = 'exp_replay.pkl'

lr = 1e-4
batch_size = 32
device = torch.device('cpu')
timesteps_per_epoch = 8

refresh_target_network_freq = 5000
loss_freq = 50
eval_freq = 5000

def make_env():
    return TetrisGame(width=12, height=26)

env = make_env()
agent = TetrisAgent(ncols=12, nrows=26, nactions=4, epsilon=0.3).to(device)

# exp_replay = ReplayBuffer(size=1000)
with open(exp_replay_save_path, 'rb') as f:
    exp_replay = pickle.load(f)

opt = torch.optim.Adam(agent.parameters(), lr=lr)

target_network = TetrisAgent(ncols=12, nrows=26, nactions=4).to(device)
target_network.load_state_dict(agent.state_dict())

mean_rw_history = []
td_loss_history = []
initial_state_v_history = []

def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')

# train loop
state = env.reset()

step = 0
try:
    for step in trange(step, step+10000):

        # play
        _, state = T.play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)

        loss = T.compute_td_loss(device,
                                 obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,
                                 agent, target_network,
                                 gamma=0.99, check_shapes=False)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            mean_rw_history.append(T.evaluate(
                make_env(), agent, n_games=3, greedy=True, t_max=1000, seed=step)
            )
            initial_state_q_values = agent.get_qvalues(make_env().reset(seed=step)[np.newaxis])
            initial_state_v_history.append(np.max(initial_state_q_values))

            # clear_output(True)
            print("buffer size = %i, epsilon = %.5f" % (len(exp_replay), agent.epsilon))

            plt.figure(figsize=[16, 9])

            plt.subplot(2, 2, 1)
            plt.title("Mean reward per episode")
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(td_loss_history[-1])
            plt.subplot(2, 2, 2)
            plt.title("TD loss history (smoothened)")
            plt.plot(smoothen(td_loss_history))
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.title("Initial state V")
            plt.plot(initial_state_v_history)
            plt.grid()

            plt.show()

except KeyboardInterrupt:
    # Обрабатываем Ctrl+C
    pass

torch.save(agent.state_dict(), agent_save_path)
print(f'Веса нейронной сети сохранены в {agent_save_path}')
