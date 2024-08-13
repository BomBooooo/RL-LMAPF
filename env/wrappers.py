import datetime
import numpy as np
import time
import uuid
import torch
import torch.nn.functional as F

import gymnasium
from typing import Any, SupportsFloat

from env.Render import Render


class RenderWrapper(gymnasium.Wrapper):
    SHELF = "rgb(0,127,0)"
    OBSTACLE = "rgb(0,0,0)"
    DELIVERY_POSITION = "rgb(184,227,254)"

    def __init__(
        self,
        env: gymnasium.Env,
        render_cell_size: int,
        svgdir,
        time_interval: float = 1,
    ):
        super().__init__(env)
        self._cell = render_cell_size
        self.colors = (self.SHELF, self.OBSTACLE)
        self.time_interval = time_interval
        self.svgdir = svgdir

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        returns = super().reset(seed=seed, options=options)
        self._render = Render(
            self.env.unwrapped.w,
            self.env.unwrapped.h,
            self._cell,
            self.colors,
            self.time_interval,
            seed,
        )
        self._render.init_draw(
            self.agent.pos,
            self.agent.goal,
            self.env.unwrapped.map.obstacle_set,
            self.env.unwrapped.map.shelves_set,
            self.env.max_duration,
        )
        return returns

    def render(self):
        super().render()
        duration = self.env.duration
        self._render.update(
            duration, self.unwrapped.agent.pos, self.unwrapped.agent.goal
        )

    def save_svg(self, name):
        self._render.save(self.svgdir / name, self.env.duration)


class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        obs_radius: int,
        nbr_radius: int,
        obs_dim: int,
        isCentralized=False,
        device="cpu",
        dtype=torch.int64,
    ):
        super().__init__(env)
        super(ObservationWrapper, self).__init__(env)
        self._obs_radius = obs_radius
        self.nbr_radius = nbr_radius
        self._obs_dim = obs_dim
        self.obs_size = self._obs_radius * 2 + 1
        self.observation_space = gymnasium.spaces.Box(
            low=-1,
            high=self.obs_size**2,
            shape=(self._obs_dim, self.obs_size, self.obs_size),
            dtype=np.float32,
        )
        self.pad = (self._obs_radius,) * 4
        self.device = device
        self.dtype = dtype
        self._pad_values = (self.unwrapped.map.OBSTACLE, self.unwrapped.map.OBSTACLE)
        self._isCentralized = isCentralized

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        return (None, info) if self._isCentralized else (self._all_obs(), info)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        _, rewards, terminated, truncated, output = self.env.step(action)
        obs = None if self._isCentralized else self._all_obs()
        return obs, rewards, terminated, truncated, output

    def dis2idx(self, dis):
        # return ((dis[1] + self._obs_radius) * (self._obs_radius * 2 + 1)) + (
        #     dis[0] + self._obs_radius
        # )
        # 2r2 + (2 * dis[1] + 2)r + dis[1] + dis[0]
        # assert (
        #     dis[0] <= self._obs_radius
        #     and dis[1] <= self._obs_radius
        #     and dis[0] >= -self._obs_radius
        #     and dis[1] >= -self._obs_radius
        # )
        return (
            2 * self.nbr_radius**2
            + 2 * (dis[1] + 1) * self.nbr_radius
            + dis[1]
            + dis[0]
        )

    def neighbors(self, pos_start):
        distance = pos_start.unsqueeze(1) - pos_start.unsqueeze(0)
        neighbor_idx = torch.where(
            (distance[:, :, 0] <= self.nbr_radius)
            & (distance[:, :, 0] >= -self.nbr_radius)
            & (distance[:, :, 1] <= self.nbr_radius)
            & (distance[:, :, 1] >= -self.nbr_radius)
        )
        neighbor_dict = {}
        for key, value in zip(neighbor_idx[0], neighbor_idx[1]):
            key = key.item()
            value = value.item()
            if key == value:
                continue
            neighbor_dict.setdefault(key, ([], []))
            neighbor_dict[key][0].append(value)
            idx = self.dis2idx(distance[key, value])
            assert idx not in neighbor_dict[key][1]
            neighbor_dict[key][1].append(idx)
        return {
            k: (
                torch.tensor(v[0], device=self.device, dtype=torch.int32),
                torch.tensor(v[1], device=self.device, dtype=torch.int32),
            )
            for k, v in neighbor_dict.items()
        }

    def tensors(self, agent_set):
        map_tensor = F.pad(
            agent_set.map_in - 1,
            self.pad,
            "constant",
            value=self.unwrapped.map.OBSTACLE,
        )
        cost_tensor = F.pad(
            agent_set.h_map, self.pad, "constant", value=agent_set.h_map.max()
        )
        agent_map = F.pad(
            agent_set.agent_map,
            self.pad,
            "constant",
            value=self.unwrapped.map.FREE,
        )
        env_map = torch.stack([map_tensor, agent_map], dim=0)
        return env_map, cost_tensor

    def _all_obs(self):
        agent_set = self.unwrapped.agent
        pos_start = agent_set.pos
        pos_end = agent_set.pos + self.obs_size

        neighbor_dict = self.neighbors(pos_start)
        env_map, cost_tensor = self.tensors(agent_set)
        map_obs = []
        cost_obs = []
        for agent_id in range(self.unwrapped.agent_num):
            map_obs.append(
                env_map[
                    :,
                    pos_start[agent_id, 0] : pos_end[agent_id, 0],
                    pos_start[agent_id, 1] : pos_end[agent_id, 1],
                ]
            )
            cost_obs.append(
                cost_tensor[agent_id][
                    pos_start[agent_id, 0] : pos_end[agent_id, 0],
                    pos_start[agent_id, 1] : pos_end[agent_id, 1],
                ]
            )
        map_obs = torch.stack(map_obs)
        cost_obs = torch.stack(cost_obs).unsqueeze(1)
        cost_obs -= cost_obs.min(-1, True)[0].min(-2, True)[0]
        cost_obs[map_obs[:, 0:1] == -1] = -1
        return torch.concat([map_obs, cost_obs], dim=1).float(), neighbor_dict


class TimeLimit(gymnasium.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self.max_duration = duration
        self.duration = None

    def step(self, action):
        obs, rewards, terminated, _, info = self.env.step(action)
        self.duration += 1
        if self.duration >= self.max_duration:
            info["success_rate"], info["closer_rate"] = self._cal_rate()
            info["duration"] = self.duration
            truncated = True
        else:
            truncated = False
        info["total_time"] = time.time() - self.time
        return obs, rewards, terminated, truncated, info

    def _cal_rate(self):
        total_num = self.unwrapped.agent_num * self.duration
        collision_num = (
            self.unwrapped.metrics.get("change")
            + self.unwrapped.metrics.get("cell")
            + self.unwrapped.metrics.get("border")
            + self.unwrapped.metrics.get("obstacle")
        )
        success_rate = np.around(1 - (collision_num / total_num), 4)
        closer_rate = np.around(
            (self.unwrapped.metrics.get("closer") / total_num),
            4,
        )
        return success_rate, closer_rate

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        returns = super().reset(seed=seed, options=options)
        self.duration = 0
        self.time = time.time()
        return returns


class ActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [env.get_wrapper_attr("action_dim")] * self.unwrapped.agent_num
        )

    def step(
        self, action: np.array
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert len(action) == self.unwrapped.agent_num
        if len(action.shape) == 2:
            assert action.shape[1] == 1, f"Invalid action shape {action.shape}"
            action = action.squeeze(-1)
        return self.env.step(action)


class UUID(gymnasium.Wrapper):

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return super().reset(seed=seed, options=options)
