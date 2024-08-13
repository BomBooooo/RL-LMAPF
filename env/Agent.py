import torch
import random
import numpy as np
from collections import deque

from env.Point import Point
from policy.RL_LMAPF import RL_LMAPF

class ReplayBuffer:
    def __init__(
        self,
        obs_dim,
        radius,
        num_1d,
        agent_num,
        maxlen=1e6,
        batch_size=16,
        seed=None,
        device="cpu",
    ) -> None:
        self.buffer = []
        size = radius * 2 + 1
        self.buffer_3d = torch.zeros(
            (int(maxlen), obs_dim * 2, size, size), dtype=torch.float32
        )
        self.buffer_1d = np.zeros((int(maxlen), num_1d), dtype=np.float32)
        self.agent_num = agent_num
        self.size = 0
        self.head = 0
        self.maxlen = int(maxlen)
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)

    def isFull(self):
        return self.size == self.maxlen

    def add1(self, trans_3d, neighbor, trans_1d):
        if self.size == self.maxlen:
            self.buffer[self.head] = neighbor
            self.buffer_3d[self.head] = trans_3d
            self.buffer_1d[self.head] = trans_1d
            self.head = (self.head + 1) % self.maxlen
        else:
            self.buffer.append(neighbor)
            self.buffer_3d[self.size] = trans_3d
            self.buffer_1d[self.size] = trans_1d
            self.size += 1

    def add(self, obs, actions, reward, next_obs, done):
        obs, neighbors = obs
        next_obs, next_neighbors = next_obs
        trans_3d = torch.concat([obs, next_obs], dim=1).cpu()
        trans_1d = np.stack([actions, reward, done * np.ones_like(reward)], axis=-1)
        for i in range(self.agent_num):
            n = (
                None
                if neighbors.get(i) is None
                else (obs[neighbors[i][0]].cpu(), neighbors[i][1].cpu())
            )
            new_n = (
                None
                if next_neighbors.get(i) is None
                else (next_obs[next_neighbors[i][0]].cpu(), next_neighbors[i][1].cpu())
            )
            neighbor = (n, new_n)
            self.add1(trans_3d[i], neighbor, trans_1d[i])

    def sample(self, batch_size=None):
        while True:
            idx = self.rng.integers(
                0, self.size, size=self.batch_size if batch_size is None else batch_size
            )
            obs, next_obs = torch.chunk(self.buffer_3d[idx].to(self.device), 2, dim=1)
            actions, reward, done = torch.chunk(
                torch.tensor(self.buffer_1d[idx], device=self.device), 3, dim=1
            )
            neighbors = []
            next_neighbors = []
            for i in idx:
                neighbor, next_neighbor = self.buffer[i]
                if neighbor is not None:
                    neighbors.append(
                        (neighbor[0].to(self.device), neighbor[1].to(self.device))
                    )
                else:
                    neighbors.append(None)
                if next_neighbor is not None:
                    next_neighbors.append(
                        (
                            next_neighbor[0].to(self.device),
                            next_neighbor[1].to(self.device),
                        )
                    )
                else:
                    next_neighbors.append(None)
            yield (
                obs,
                neighbors,
                actions.long(),
                reward,
                next_obs,
                next_neighbors,
                done,
            )


class AgentSet:
    def __init__(
        self,
        args,
        map_in,
        agent_num,
        radius=None,
        device="cpu",
        dtype=torch.int32,
        seed=None,
        isLoad=False,
        version="v1",
    ):
        self.agent_num = agent_num
        self.radius = radius * 2 + 1 if radius else radius
        self.dtype = dtype
        self.device = device
        self.map_file = args.map_file
        self.map_in = torch.tensor(map_in, dtype=self.dtype, device=self.device) + 1

        obstacle = torch.where(self.map_in == 0)
        self.obstacle = torch.tensor(
            list(zip(obstacle[0], obstacle[1])), device=self.device, dtype=self.dtype
        )
        self.neighbors = torch.tensor(
            [[0, 1], [0, -1], [1, 0], [-1, 0]], device=self.device, dtype=self.dtype
        )
        self.rl_lmapf = RL_LMAPF(
            args.obs_dim,
            args.obs_radius,
            args.nbr_radius,
            args.hidden_dim,
            args.action_dim,
            args.msg_dim,
            args.activation,
            args.norm,
            args.dropout_p,
            args.lr,
            args.eps,
            args.wd,
            args.gamma,
            args.epsilon,
            args.target_update,
            device,
            seed=None,
            isLoad=isLoad,
        )
        self.rl_lmapf.build()
        self.rng = np.random.default_rng(seed)
        self.large_num = 1e6
        self.step = 0

    def state_dict(self):
        return self.rl_lmapf.q_net.state_dict(), self.rl_lmapf.target_q_net.state_dict()

    def load_state_dict(self, state_dict):
        self.rl_lmapf.q_net.load_state_dict(state_dict[0])
        self.rl_lmapf.target_q_net.load_state_dict(state_dict[1])

    def save(self, step, reward, path):
        self.rl_lmapf.save(step, reward, path)

    def generate_agent(self, free_set):
        select_pos = self.rng.choice(list(free_set), size=self.agent_num, replace=False)
        return torch.tensor(select_pos, dtype=self.dtype, device=self.device)

    def reset(self, free_set, goals, assigned_goals):
        self.agent_map = torch.zeros_like(
            self.map_in, dtype=self.dtype, device=self.device
        )
        assert self.agent_num < len(free_set), "agent_num is larger than free spaces."
        self.pos = self.generate_agent(free_set)
        for i in range(self.agent_num):
            self.agent_map[self.pos[i, 0], self.pos[i, 1]] = 1
        self.goal = self.assign_goal2agent(self.agent_num, goals, assigned_goals)
        self.h_map = self.cal_h_map(self.goal)
        self.update_h_tensor = torch.zeros(
            (self.agent_num,), device=self.device, dtype=self.dtype
        )
        for i in range(self.agent_num):
            assert (
                self.h_map[i, self.goal[i, 0], self.goal[i, 1]] != -1
            ), f"{self.map_file}: start({self.pos[i]}) to goal({self.goal[i]}) not connected"

    def get_agent_position(self, agent_id):
        return Point(self.pos[agent_id].cpu().numpy())

    def move_agent(self, agent_idx, direction):
        agent_idx = torch.tensor(agent_idx, device=self.device, dtype=self.dtype)
        h = self.h_map[agent_idx, self.pos[agent_idx, 0], self.pos[agent_idx, 1]]
        self.agent_map[self.pos[agent_idx, 0], self.pos[agent_idx, 1]] -= 1
        self.pos[agent_idx] += direction.to_tensor(self.device, self.dtype)
        self.agent_map[self.pos[agent_idx, 0], self.pos[agent_idx, 1]] += 1
        new_h = self.h_map[agent_idx, self.pos[agent_idx, 0], self.pos[agent_idx, 1]]
        on_goal = torch.sum(self.pos[agent_idx] == self.goal[agent_idx]).item()
        if on_goal == 2:
            self.update_h_tensor[agent_idx] = 1
            return "finish"
        if new_h > h:
            return "further"
        if new_h < h:
            return "closer"
        raise ValueError(
            f"agent_idx: {agent_idx}; h:{h},{new_h}; pos:{self.pos[agent_idx]}; goal: {self.goal[agent_idx]}"
        )

    def update(self, goals, assigned_goals):
        assert self.agent_map.max() == 1
        update_num = self.update_h_tensor.sum()
        if update_num == 0:
            return
        update_h_id = self.update_h_tensor == 1
        finished_goals = self.goal[update_h_id]
        self.goal[update_h_id] = self.assign_goal2agent(
            update_num, goals, assigned_goals, finished_goals
        )
        self.h_map[update_h_id] = self.cal_h_map(self.goal[update_h_id])
        self.update_h_tensor[:] = 0
        self.step += 1

    def assign_goal2agent(self, num: int, goals, assigned_goals, finished_goals=None):
        agent_goals = []
        if finished_goals is not None:
            finished_goals = finished_goals.cpu()
            for i in range(finished_goals.shape[0]):
                assigned_goals.remove(tuple(finished_goals[i].numpy()))
        for _ in range(num):
            if goals:
                goal = goals.popleft()
                agent_goals.append(goal)
                assigned_goals.add(goal)
            else:
                print("No tasks to assign.")
                return None
        return torch.tensor(agent_goals, device=self.device, dtype=self.dtype)

    def cal_h_map(self, goal):
        goal_num = goal.shape[0]
        h_map = (self.map_in - 1) * torch.ones(
            (goal_num,) + self.map_in.shape, device=self.device, dtype=self.dtype
        )
        update_map = h_map.clone()
        start = torch.unsqueeze(goal, dim=1)
        agent_range = torch.arange(goal_num, device=self.device, dtype=self.dtype)
        z = (
            agent_range.unsqueeze(-1)
            * torch.ones(
                (goal_num, self.neighbors.shape[0]),
                device=self.device,
                dtype=self.dtype,
            )
        ).flatten()
        dis = 1
        while True:
            expend = (start + self.neighbors).reshape(-1, 2)
            illegal = torch.where(
                (expend[:, 0] < 0)
                | (expend[:, 0] >= self.map_in.shape[0])
                | (expend[:, 1] < 0)
                | (expend[:, 1] >= self.map_in.shape[1])
            )
            expend[illegal] = self.obstacle[0]
            update_map[(z, expend[:, 0], expend[:, 1])] = dis
            h_map = torch.where(h_map == 0, update_map, h_map)
            start = torch.where(h_map == dis)
            z = (
                (
                    start[0]
                    * torch.ones(
                        (self.neighbors.shape[0],) + start[0].shape,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                .transpose(0, 1)
                .flatten()
            )
            start = torch.stack(start[1:], dim=1).type(self.dtype).unsqueeze(1)
            if start.shape[0] == 0:
                h_map[h_map == -1] = self.large_num
                assert torch.where(h_map == 0)[0].shape[0] == 0, torch.where(h_map == 0)
                h_map[agent_range, goal[:, 0], goal[:, 1]] = 0
                return h_map
            dis += 1

    @torch.no_grad()
    def policy(self, state, epsilon=None):
        return self.rl_lmapf.policy(state, self.step, epsilon)

    def train_model(self, batch_size):
        return self.rl_lmapf.update(batch_size)
