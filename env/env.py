import torch
import pathlib
import gymnasium
import numpy as np
from copy import deepcopy
from typing import Any, SupportsFloat

from env.Reward import Reward
from env.Metrics import Metrics
from env.GridMap import GridMap
from env.Point import Point


class GridEnv(gymnasium.Env):
    INFO = [
        "border",
        "obstacle",
        "shelf",
        "cell",
        "change",
        "stay",
        "closer",
        "further",
        "finish",
    ]

    def __init__(
        self,
        action_dim: int,
        file: pathlib.Path,
        agent_num: int,
        seed=None,
    ):
        self.map = GridMap(file)
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [
                action_dim,
            ]
            * agent_num,
            dtype=np.int8,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=0, shape=(self.map.w, self.map.h), dtype=np.float32
        )

    @property
    def h(self):
        return self.map.h

    @property
    def w(self):
        return self.map.w

    @property
    def obstacle_set(self):
        return self.map.obstacle_set

    @property
    def map_np(self):
        return self.map.map_np

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.map.reset(self.agent_num)
        self.agent = options["agent"]
        self.agent.reset(self.map.free_set, self.map.goals, self.map.assigned_goals)
        self.reward = Reward(self.INFO)
        self.metrics = Metrics(self.INFO)
        return None, {}

    def step(self, actions):
        rewards = np.zeros(self.agent_num)
        step_action = deepcopy(actions)
        info = {}
        used_cells, used_edges = self.occupy_cells_edges(step_action)
        stayed_agent = self.change_collision_detection(
            step_action, used_cells, used_edges, rewards, info
        )
        stayed_agent = self.cell_collision_detection(
            step_action, used_cells, stayed_agent, rewards, info
        )
        self.move(step_action, stayed_agent, rewards, info)
        self.metrics.num_check(self.agent_num)
        self.check_enough_goals()

        output = {"info": info}

        return None, rewards, False, False, output

    def render(self):
        pass

    def check_enough_goals(self):
        if len(self.map.goals) < self.agent_num:
            self.map.generate_goal(self.agent_num - len(self.map.goals))

    def write_info(self, agent_idx, info_type, rewards, info):
        rewards[agent_idx] += self.reward.get_reward(info_type)
        assert info.get(agent_idx) is None, f"info {info} already exists."
        info[agent_idx] = info_type
        self.metrics.add_scale(info_type, 1)

    def occupy_cells_edges(self, step_action):
        used_cells, used_edges = {}, {}
        for agent_idx in range(self.agent_num):
            pos = self.agent.get_agent_position(agent_idx)
            d_pos = self.action2dir(step_action[agent_idx])
            new_pos = pos + d_pos
            used_cells.setdefault(new_pos, []).append(agent_idx)
            assert (
                len(used_cells.get(new_pos)) <= 5
            ), f"One cell {new_pos} is occupied by more than five agents."
            if not d_pos.isStay():
                used_edges.setdefault((pos, new_pos), []).append(agent_idx)
                used_edges.setdefault((new_pos, pos), []).append(agent_idx)
                assert (
                    len(used_edges.get((pos, new_pos))) <= 2
                ), f"One edge {(pos, new_pos)} is occupied by more than two agents."
                assert (
                    len(used_edges.get((new_pos, pos))) <= 2
                ), f"One edge {(new_pos, pos)} is occupied by more than two agents."
        return used_cells, used_edges

    def change_collision_detection(
        self, step_action, used_cells, used_edges, rewards, info
    ):
        # 智能体换位检测
        stayed_agent = set()
        for agent_idx in range(self.agent_num):
            if agent_idx in stayed_agent:
                continue
            d_pos = self.action2dir(step_action[agent_idx])
            if d_pos.isStay():
                stayed_agent.add(agent_idx)
                self._stay(agent_idx, stayed_agent, step_action, "stay", rewards, info)
                continue
            pos = self.agent.get_agent_position(agent_idx)
            new_pos = pos + d_pos
            if len(used_edges[(pos, new_pos)]) > 1:
                changed_agents = used_edges.get((pos, new_pos))
                a0, a1 = changed_agents
                if a0 != agent_idx:
                    a0, a1 = a1, a0
                self._revert_cell(a0, used_cells, pos, new_pos)
                self._revert_cell(a1, used_cells, new_pos, pos)
                self._stay(a0, stayed_agent, step_action, "change", rewards, info)
                self._stay(a1, stayed_agent, step_action, "change", rewards, info)
                del used_edges[(pos, new_pos)]
                del used_edges[(new_pos, pos)]
        return stayed_agent

    def cell_collision_detection(
        self, step_action, used_cells, stayed_agent, rewards, info
    ):
        # 检测智能体碰撞
        for agent_idx in range(self.agent_num):
            if agent_idx in stayed_agent:
                continue
            pos = self.agent.get_agent_position(agent_idx)
            d_pos = self.action2dir(step_action[agent_idx])
            new_pos = pos + d_pos
            # 智能体保持原位
            if d_pos.isStay():
                self._stay(agent_idx, stayed_agent, step_action, "stay", rewards, info)
            # 边界检测
            elif self.map.is_out_of_border(new_pos):
                stayed_agent = self._revert_action(
                    agent_idx,
                    stayed_agent,
                    step_action,
                    used_cells,
                    pos,
                    new_pos,
                    rewards,
                    info,
                    "border",
                )
            # 障碍物碰撞检测
            elif self.map.is_obstacle(new_pos):
                stayed_agent = self._revert_action(
                    agent_idx,
                    stayed_agent,
                    step_action,
                    used_cells,
                    pos,
                    new_pos,
                    rewards,
                    info,
                    "obstacle",
                )
            # 货架碰撞检测
            elif self.map.is_shelf(new_pos):
                stayed_agent = self._revert_action(
                    agent_idx,
                    stayed_agent,
                    step_action,
                    used_cells,
                    pos,
                    new_pos,
                    rewards,
                    info,
                    "shelf",
                )
            # 智能体重合检测
            elif len(used_cells.get(new_pos)) > 1:
                assert (
                    len(used_cells.get(new_pos)) <= 5
                ), f"One cell {new_pos} is occupied by more than five agents."
                stayed_agent = self._revert_action(
                    agent_idx,
                    stayed_agent,
                    step_action,
                    used_cells,
                    pos,
                    new_pos,
                    rewards,
                    info,
                    "cell",
                )
        return stayed_agent

    def move(self, step_action, stayed_agent, rewards, info):
        for agent_idx in range(self.agent_num):
            if agent_idx in stayed_agent:
                continue
            direction = self.action2dir(step_action[agent_idx])
            agent_info = self.agent.move_agent(agent_idx, direction)
            self.write_info(agent_idx, agent_info, rewards, info)

    def _stay(self, agent_id, stayed_agent, step_action, info_type, rewards, info):
        stayed_agent.add(agent_id)
        step_action[agent_id] = 0
        self.write_info(agent_id, info_type, rewards, info)

    def _revert_cell(self, agent_idx, used_cells, pos, new_pos):
        used_cells.get(new_pos).remove(agent_idx)
        used_cells.setdefault(pos, []).append(agent_idx)

    def _revert_action(
        self,
        agent_idx,
        stayed_agent,
        step_action,
        used_cells,
        pos,
        new_pos,
        rewards,
        info,
        info_type="cell",
    ):
        stayed_agent.add(agent_idx)
        step_action[agent_idx] = 0
        self._revert_cell(agent_idx, used_cells, pos, new_pos)
        self.write_info(agent_idx, info_type, rewards, info)
        cell = used_cells.get(pos)
        if len(cell) > 1:
            for a in cell[:-1]:
                a_pos = self.agent.get_agent_position(a)
                a_new_pos = a_pos + self.action2dir(step_action[a])
                assert a_new_pos == pos, f"a_new_pos {a_new_pos} != pos {pos}"
                stayed_agent = self._revert_action(
                    a,
                    stayed_agent,
                    step_action,
                    used_cells,
                    a_pos,
                    a_new_pos,
                    rewards,
                    info,
                )
        return stayed_agent

    def metrics_result(self):
        return self.metrics.total_metrics

    @staticmethod
    def action2dir(action):
        # checking_table = {0: Point((0, 0)), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
        checking_table = {
            0: Point((0, 0)),
            1: Point((0, 1)),
            2: Point((1, 0)),
            3: Point((0, -1)),
            4: Point((-1, 0)),
        }
        return checking_table[action]
