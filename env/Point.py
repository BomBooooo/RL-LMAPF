import numpy as np
import torch


class Point:
    def __init__(self, pos):
        if isinstance(pos, np.ndarray):
            self.pos = pos
        elif isinstance(pos, (tuple, list)):
            self.pos = np.array(pos)
        elif isinstance(pos, torch.Tensor):
            self.pos = pos.cpu().numpy()
        else:
            raise ValueError(f"Error value: {pos}")

    def __hash__(self) -> int:
        return hash(tuple(self.pos))

    def __eq__(self, o) -> bool:
        return np.array_equal(self.pos, o.pos)

    def __repr__(self) -> str:
        return f"Point({self.pos})"

    def __add__(self, o):
        if isinstance(o, Point):
            return Point(self.pos + o.pos)
        if isinstance(o, np.ndarray):
            return Point(self.pos + o)

    def __sub__(self, o):
        return Point(self.pos - o.pos)

    def __getitem__(self, idx):
        assert idx in (0, 1), f"Invalid index: {idx}"
        return self.pos[idx]
    
    def to_tensor(self, device="cpu", dtype=torch.int32):
        return torch.from_numpy(self.pos).to(device).type(dtype)

    def isStay(self):
        return np.array_equal(self.pos, np.array([0, 0]))

    def to_action(self):
        pos = tuple(self.pos)
        action_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
        assert pos in action_table, f"Invalid direction: {pos}"
        return action_table[pos]
