import torch
import time
import datetime
import pathlib
import numpy as np

from env.env import GridEnv
from env.Agent import AgentSet, ReplayBuffer
from env.wrappers import ObservationWrapper, TimeLimit, RenderWrapper
from tools import Logger


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.train_env = self.create_env()
        self.eval_env = self.create_env(isRender=args.isRender)
        self.train_logger = Logger(args.logdir)
        self.eval_logger = Logger(args.evaldir)

        self.finish_num = 0

        self.agent = self.create_agent()
        print(
            f"Optimizer RL-LMAPF has {sum(param.numel() for param in self.agent.rl_lmapf.parameters())} variables."
        )
        self.buffer = ReplayBuffer(
            args.obs_dim,
            args.obs_radius,
            3,
            args.agent_num,
            args.buffer_size,
            args.batch_size,
            args.seed,
            args.device,
        )
        self.eval_agent = self.create_agent()

    def create_agent(self):
        return AgentSet(
            self.args,
            self.train_env.unwrapped.map_np,
            self.args.agent_num,
            radius=self.args.obs_radius,
            device=self.args.device,
            dtype=self.args.dtype,
            seed=self.args.seed,
            isLoad=self.args.isLoad,
        )

    def create_env(self, isRender=False):
        env = GridEnv(
            action_dim=self.args.action_dim,
            file=pathlib.Path(self.args.map_file),
            agent_num=self.args.agent_num,
            seed=self.args.seed,
        )
        env = TimeLimit(env, duration=self.args.time_limit)
        env = ObservationWrapper(
            env,
            self.args.obs_radius,
            self.args.nbr_radius,
            self.args.obs_dim,
            device=self.device,
        )
        if isRender:
            return RenderWrapper(
                env,
                self.args.render_cell_size,
                self.args.svgdir,
                self.args.time_interval,
            )
        return env

    def __call__(self, isEval=False):
        if isEval:
            for i in range(self.args.eval_episode_num):
                self.eval(steps=i)
        else:
            while True:
                obs, info = self.train_env.reset(
                    seed=self.args.seed, options={"agent": self.agent}
                )
                while True:
                    actions = self.agent.policy(obs)
                    next_obs, reward, terminated, truncated, info = self.train_env.step(
                        actions
                    )
                    self.agent.update(
                        self.train_env.unwrapped.map.goals,
                        self.train_env.unwrapped.map.assigned_goals,
                    )
                    done = torch.tensor(terminated or truncated, dtype=torch.bool)
                    self.buffer.add(obs, actions, reward, next_obs, done)
                    obs = next_obs
                    finish = self.train()
                    if finish:
                        return
                    if terminated or truncated:
                        obs, reset_info = self.train_env.reset(
                            seed=self.args.seed, options={"agent": self.agent}
                        )

    def train(self):
        if not hasattr(self, "dataset"):
            if self.buffer.isFull():
                self.dataset = self.buffer.sample(self.args.batch_size)
                print("Start Training...")
        else:
            if self.train_logger.step % self.args.eval_every == 0:
                self.eval()
            t0 = time.time()
            data = next(self.dataset)
            t1 = time.time()
            loss = self.agent.train_model(data)
            if self.train_logger.step % self.args.log_every == 0:
                self.train_logger.scalar("loss", loss)
                self.train_logger.scalar("data_time", round(t1 - t0, 3))
                self.train_logger.scalar("train_time", round(time.time() - t1, 3))
                self.train_logger.write(desc="train")
            self.train_logger.step += 1
            if self.train_logger.step > self.args.steps:
                return True

    def eval(self, steps=0):
        eval_step = self.train_logger.step + steps
        print(f"Start {eval_step} Evaluation ...")
        t0 = time.time()
        log_dict = {
            "search_time": 0,
            "update_time": 0,
            "interact_time": 0,
            "render_time": 0,
            "reward": 0,
        }
        self.eval_agent.load_state_dict(self.agent.state_dict())
        obs, info = self.eval_env.reset(
            seed=self.args.seed, options={"agent": self.eval_agent}
        )
        while True:
            t1 = time.perf_counter()
            actions = self.eval_agent.policy(obs, epsilon=0)
            t2 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            t3 = time.perf_counter()
            self.eval_agent.update(
                self.eval_env.unwrapped.map.goals,
                self.eval_env.unwrapped.map.assigned_goals,
            )
            t4 = time.perf_counter()
            if self.args.isRender:
                self.eval_env.render()
            t5 = time.perf_counter()
            log_dict["reward"] += np.mean(reward)
            log_dict["search_time"] += t2 - t1
            log_dict["update_time"] += t4 - t3
            log_dict["interact_time"] += t3 - t2
            log_dict["render_time"] += t5 - t4
            obs = next_obs
            if terminated or truncated:
                metrics = self.eval_env.unwrapped.metrics_result()
                log_dict = {k: v / self.eval_env.duration for k, v in log_dict.items()}
                log_dict["total_time"] = info["total_time"]
                metrics["reward"] = np.around(log_dict["reward"], 5)
                metrics["closer_rate"] = log_dict["closer_rate"] = info["closer_rate"]
                metrics["success_rate"] = info["success_rate"]
                log_dict["throughput"] = metrics["finish"] / self.args.time_limit
                [self.eval_logger.scalar(k, v) for k, v in log_dict.items()]
                self.eval_logger.save_env_metrics(eval_step, metrics, self.args.evaldir)
                self.eval_logger.write(desc="eval")
                self.eval_logger.step += 1
                r = round(metrics["reward"], 4)
                t = datetime.datetime.now().strftime("%d%H%M")
                self.eval_agent.save(
                    eval_step,
                    metrics["reward"],
                    self.args.modeldir / f"{t}_{eval_step}_reward_{r}.pth",
                )
                if self.args.isRender:
                    try:
                        self.eval_env.save_svg(f"{self.eval_logger.step}_{r}.svg")
                    except Exception as e:
                        print(f"发生了其他异常: {e}")
                        print("Cannot Save svg !!!")
                print(f"Evaluating for {time.time() - t0} s")
                return metrics["reward"]
