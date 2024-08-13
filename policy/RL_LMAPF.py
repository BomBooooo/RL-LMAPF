import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from policy import nets


class MsgWeightNet(nn.Module):
    def __init__(self, activation, norm, dropout_p):
        super().__init__()
        self.msg_weight = nets.MLP(
            2,
            [16],
            1,
            activation,
            norm=norm,
            dropout_p=dropout_p,
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        return self.softmax(self.msg_weight(x)).transpose(1, 0)


class QNet(nn.Module):
    def __init__(
        self,
        channel_in,
        radius,
        nbr_radius,
        hidden_dim,
        action_dim,
        msg_dim,
        activation,
        norm,
        dropout_p,
        device,
    ):
        super().__init__()
        self.nbr_radius = nbr_radius
        nbr_size = 2 * nbr_radius + 1
        relative_pos = torch.arange(
            -nbr_radius, nbr_radius + 1, device=device
        ) * torch.ones((nbr_size, nbr_size), device=device)
        self.relative_pos = (
            torch.stack([relative_pos, relative_pos.T]).reshape(2, -1).transpose(1, 0)
        )
        obs_dim = action_dim * 64
        middle_dim = action_dim * 32
        self.msg_dim = msg_dim
        self.nets = nn.Sequential(
            nets.CNN(
                channel_in,
                [hidden_dim * 2, hidden_dim * 4],
                hidden_dim * 8,
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                activation,
                norm=norm,
                post_activation=True,
            ),
            nn.Flatten(),
            nets.MLP(
                (hidden_dim * 8),
                [512, 512],
                obs_dim,
                activation,
                norm=norm,
                dropout_p=dropout_p,
            ),
        )
        self.msg_encoder = nets.MLP(
            obs_dim,
            [msg_dim, msg_dim * 2, msg_dim * 2],
            msg_dim,
            activation,
            norm=norm,
            dropout_p=dropout_p,
        )
        self.msg_weight = MsgWeightNet(activation, norm, dropout_p)
        self.msg_decoder = nets.MLP(
            msg_dim,
            [msg_dim * 2, msg_dim * 2, msg_dim],
            obs_dim,
            activation,
            norm=norm,
            dropout_p=dropout_p,
        )
        self.fc_A = nets.MLP(
            obs_dim * 2,
            [middle_dim, int(middle_dim / 2)],
            action_dim,
            activation,
            norm=norm,
            dropout_p=dropout_p,
        )
        self.fc_V = nets.MLP(
            obs_dim * 2,
            [middle_dim, int(middle_dim / 2)],
            1,
            activation,
            norm=norm,
            dropout_p=dropout_p,
        )

    def forward(self, state):
        obs, neighbors = state
        x = self.nets(obs)
        weight = self.msg_weight(self.relative_pos)
        neighbor_msg = torch.zeros((obs.shape[0], self.msg_dim), device=obs.device)
        if isinstance(neighbors, dict):
            others_msgs = self.msg_encoder(x)
            for i, neighbor in neighbors.items():
                if neighbor is None:
                    continue
                neighbor_msg[i] = torch.mm(
                    weight[:, neighbor[1]], others_msgs[neighbor[0]]
                )
        elif isinstance(neighbors, list):
            for i, neighbor in enumerate(neighbors):
                if neighbor is None:
                    continue
                neighbor_msg[i] = torch.mm(
                    weight[:, neighbor[1]], self.msg_encoder(self.nets(neighbor[0]))
                )
        else:
            raise ValueError("Invalid neighbors type")
        msg = self.msg_decoder(neighbor_msg)
        x = torch.cat([x, msg], dim=1)
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


class RL_LMAPF(nn.Module):
    def __init__(
        self,
        channel_in,
        radius,
        nbr_radius,
        hidden_dim,
        action_dim,
        msg_dim,
        activation,
        norm,
        dropout_p,
        lr,
        eps,
        wd,
        gamma,
        epsilon,
        target_update,
        device,
        seed=None,
        isLoad=False,
    ):
        super().__init__()
        self.q_net = (
            QNet(
                channel_in,
                radius,
                nbr_radius,
                hidden_dim,
                action_dim,
                msg_dim,
                activation,
                norm,
                dropout_p,
                device,
            )
            .to(device)
            .float()
        )
        self.target_q_net = (
            QNet(
                channel_in,
                radius,
                nbr_radius,
                hidden_dim,
                action_dim,
                msg_dim,
                activation,
                norm,
                dropout_p,
                device,
            )
            .to(device)
            .float()
        )
        if isLoad:
            if isinstance(isLoad, str):
                state_dict = torch.load(isLoad)
                self.q_net.load_state_dict(state_dict)
                self.target_q_net.load_state_dict(state_dict)
            else:
                self.q_net.load_state_dict(isLoad)
                self.target_q_net.load_state_dict(isLoad)
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=lr,
            eps=eps,
            weight_decay=wd,
        )
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.best_reward = -np.inf
        self.inp_size = radius * 2 + 1
        self.inp_channel = channel_in
        self.sync()

    def build(self):
        obs = torch.zeros(2, self.inp_channel, self.inp_size, self.inp_size).to(self.device)
        neighbors = [None, None]
        self.policy((obs, neighbors), 0, 0)

    def sync(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def state_dict(self):
        return self.q_net.state_dict()

    def policy(self, state, step, epsilon):
        if epsilon is None:
            epsilon = (self.epsilon[0] - self.epsilon[1]) * np.exp(
                -step
            ) + self.epsilon[1]
        elif epsilon == 0:
            return self.q_net(state).argmax(1).detach().cpu().numpy()
        random_action = self.rng.integers(0, self.action_dim, size=(state[0].shape[0],))
        q_action = self.q_net(state).argmax(1).detach().cpu().numpy()
        return np.where(
            self.rng.random(q_action.shape) < epsilon, random_action, q_action
        )

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def save(self, step, reward, path):
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(self.q_net.state_dict(), path)
        elif step % 10000 == 0:
            torch.save(self.q_net.state_dict(), path)

    def update(self, dataset):
        obs, neighbor, actions, reward, next_obs, next_neighbors, done = dataset
        q_values = self.q_net((obs, neighbor)).gather(1, actions)
        max_action = self.q_net((next_obs, next_neighbors)).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net((next_obs, next_neighbors)).gather(
            1, max_action
        )
        q_targets = reward + self.gamma * max_next_q_values * (1 - done)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.sync()
        self.count += 1

        return dqn_loss.item()
