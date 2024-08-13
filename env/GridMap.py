import pathlib
import numpy as np
from collections import deque

from env.Point import Point


class GridMap:
    FREE = 0
    OBSTACLE = -1
    SHELF = -2

    def __init__(self, file: pathlib.Path, seed=None) -> None:
        self.file = file
        self.read_map()
        if "warehouse" in self.file.name:
            self.shelf_np = np.zeros_like(self.map_np)
            self.shelf_np[self.map_np == self.SHELF] = 1
        self.map_np[self.map_np == self.SHELF] = self.OBSTACLE
        self.rng = np.random.default_rng(seed)

    def reset(self, agent_num: int):
        self.goals = deque()
        self.assigned_goals = set()
        every_task = 2
        self.generate_goal(agent_num * every_task)

    @property
    def goal_num(self):
        return len(self.goal_set)

    def is_out_of_border(self, new_pos):
        return (
            new_pos[0] < 0
            or new_pos[0] >= self.w
            or new_pos[1] < 0
            or new_pos[1] >= self.h
        )

    def is_obstacle(self, new_pos):
        return (new_pos[0], new_pos[1]) in self.obstacle_set

    def is_shelf(self, new_pos):
        return (new_pos[0], new_pos[1]) in self.shelves_set

    def read_map(self):
        self.free_set = set()
        self.shelves_set = set()
        self.obstacle_set = set()
        self.task_set = set()
        with open(self.file, "r") as f:
            for i, line in enumerate(f):
                if i > 3:
                    y = i - 4
                    for x, n in enumerate(line.strip()):
                        if n == "@":
                            self.map_np[x, y] = self.OBSTACLE
                            self.obstacle_set.add((x, y))
                        elif n == "T":
                            self.map_np[x, y] = self.SHELF
                            self.shelves_set.add((x, y))
                            if (
                                x != 0
                                and y != 0
                                and x != self.w - 1
                                and y != self.h - 1
                            ):
                                self.task_set.add((x, y + 1))
                                self.task_set.add((x, y - 1))
                        elif n == ".":
                            self.map_np[x, y] = self.FREE
                            self.free_set.add((x, y))
                elif i == 1:
                    self.h = int(line.strip().split(" ")[-1])
                elif i == 2:
                    self.w = int(line.strip().split(" ")[-1])
                    self.map_np = np.zeros((self.w, self.h))
        self.task_set -= self.shelves_set

    def generate_goal(self, num_goals: int):
        if "warehouse" in self.file.name:
            goal_free_set = self.task_set - self.assigned_goals - set(self.goals)
        else:
            goal_free_set = self.free_set - self.assigned_goals - set(self.goals)
        assert num_goals <= len(goal_free_set)
        goal = self.rng.choice(list(goal_free_set), size=num_goals, replace=False)
        self.goals.extend([tuple(g) for g in goal])
