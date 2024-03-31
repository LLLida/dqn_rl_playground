import numpy as np
import torch
from typing import Any, Tuple

from .replay_buffer import ReplayBuffer

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, seed=None) -> float:
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset(seed=seed)
        reward = 0
        # for _ in range(t_max):
        for iterations in range(t_max):
            qvalues = agent.get_qvalues(s[np.newaxis])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, terminated, truncated = env.step(action)
            reward += r
            if terminated or truncated:
                break

        rewards.append(reward)
    return np.mean(rewards)

def play_and_record(initial_state, agent, env, exp_replay: ReplayBuffer, n_steps=1) -> Tuple[int, Any]:
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends due to termination or truncation, add record with done=terminated and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        qvalues = agent.get_qvalues(s[np.newaxis])
        a = agent.sample_actions(qvalues)[0]

        next_s, r, terminated, truncated = env.step(a)

        sum_rewards += r

        if terminated or truncated:
            exp_replay.add(s, a, r, next_s, done=terminated)
            env.reset()
        else:
            exp_replay.add(s, a, r, next_s, done=False)
        s = next_s

    return sum_rewards, s

def compute_td_loss(device,
                    states: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_states: np.ndarray,
                    is_done: np.ndarray,
                    agent,
                    target_network,
                    gamma: float = 0.99,
                    check_shapes: bool = False):
    states = torch.tensor(states, device=device, dtype=torch.float32)    # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states) # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions] # shape: [batch_size]
    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, axis=1)[0]
    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach())**2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss

def compute_td_loss_weighted(device,
                             states: np.ndarray,
                             actions: np.ndarray,
                             rewards: np.ndarray,
                             next_states: np.ndarray,
                             is_done: np.ndarray,
                             weights: np.ndarray,
                             agent,
                             target_network,
                             gamma: float = 0.99):
    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.int64)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)

    predicted_next_qvalues = target_network(next_states) # shape: [batch_size, n_actions]

    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions] # shape: [batch_size]
    next_state_values = torch.max(predicted_next_qvalues, axis=1)[0]

    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach())**2 * weights)

    return loss
