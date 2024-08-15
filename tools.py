import torch
import numpy as np
import pandas as pd
import random
import os
import json
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def to_np(x):
    return x.detach().cpu().numpy()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_args"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Logger:
    def __init__(self, logdir, name="losses", step=0):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=10000)
        self.name = name
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._path = {}
        self._text = {}
        self._images = {}
        self._videos = {}
        self.step = step
        columns = [
            "step",
            "reward",
            "finish",
            "closer_rate",
            "success_rate",
            "border",
            "obstacle",
            "shelf",
            "cell",
            "change",
            "stay",
            "further",
            "closer",
        ]
        self.env_metrics = pd.DataFrame(columns=columns)

    def scalar(self, name, value):
        self._scalars[name] = round(float(value), 3)

    def path(self, name, path):
        self._path[name] = path

    def text(self, name, text):
        self._text[name] = text

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, step=False, desc=""):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        print(f"[{desc}-{step}]", " / ".join(f"{k} {v:.3f}" for k, v in scalars))
        with (self._logdir / f"{self.name}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        self._writer.flush()
        self._scalars = {}
        self._text = {}
        self._images = {}
        self._videos = {}


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_metrics(directory, metrics):
    filename = directory / "env_metrics.json"
    with filename.open("w") as f:
        json.dump(metrics, f, indent=4)


def save_json(data, path):
    file = str(path.resolve())
    with open(file, "w") as f:
        data = {k: str(v) for k, v in data.items()}
        json.dump(data, f)


def print_run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return result, run_time

    return wrapper


def action2dir(action):
    # checking_table = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    x = action - 1
    x = (x % 2) * ((-1) ** (x // 2))
    y = (action % 2) * ((-1) ** (action // 2))
    idx = torch.stack((x, y), axis=1)
    idx[:, 0:1][action == 0] = 0
    return idx.long()


def dir2action(direction):
    # checking_table = {(0, 0): 0,(0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    action = torch.zeros((direction.shape[0], 1)).long()
    action[torch.where(direction[:, 1] == 1)] = 1
    action[torch.where(direction[:, 0] == 1)] = 2
    action[torch.where(direction[:, 1] == -1)] = 3
    action[torch.where(direction[:, 0] == -1)] = 4
    return action
