import time
import pathlib
import numpy as np

from env.env import GridEnv
from env.Agent import AgentSet
from env.wrappers import ObservationWrapper, TimeLimit, RenderWrapper
from tools import Logger


class Evaluater:
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.eval_logger = Logger(args.evaldir)

    def create_agent(self):
        return AgentSet(
            self.args,
            self.eval_env.unwrapped.map_np,
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

    def __call__(self):
        self.eval_env = self.create_env(isRender=self.args.isRender)
        self.agent = self.create_agent()
        for i in range(self.args.eval_episode_num):
            self.eval(i)

    def eval(self, eval_step):
        print(f"Start {eval_step} Evaluation ...")
        t0 = time.time()
        log_dict = {
            "search_time": 0,
            "update_time": 0,
            "interact_time": 0,
            "render_time": 0,
            "reward": 0,
        }
        obs, info = self.eval_env.reset(
            seed=self.args.seed, options={"agent": self.agent}
        )
        while True:
            t1 = time.perf_counter()
            actions = self.agent.policy(obs, epsilon=0)
            t2 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            t3 = time.perf_counter()
            self.agent.update(
                self.eval_env.unwrapped.map.goals,
                self.eval_env.unwrapped.map.assigned_goals,
            )
            t4 = time.perf_counter()
            if self.args.isRender:
                self.eval_env.render()
            log_dict["reward"] += np.mean(reward)
            log_dict["search_time"] += t2 - t1
            log_dict["update_time"] += t4 - t3
            log_dict["interact_time"] += t3 - t2
            obs = next_obs
            if terminated or truncated:
                metrics = self.eval_env.unwrapped.metrics_result()
                log_dict = {k: v / self.eval_env.duration for k, v in log_dict.items()}
                metrics["reward"] = np.around(log_dict["reward"], 5)
                log_dict["finish"] = metrics["finish"]
                log_dict["throughput"] = metrics["finish"] / self.args.time_limit
                [self.eval_logger.scalar(k, v) for k, v in log_dict.items()]
                self.eval_logger.save_env_metrics(eval_step, metrics, self.args.evaldir)
                self.eval_logger.write(desc="eval")
                self.eval_logger.step += 1
                r = round(metrics["reward"], 4)
                if self.args.isRender:
                    try:
                        self.eval_env.save_svg(f"{self.eval_logger.step}_{r}.svg")
                    except Exception as e:
                        print(f"发生了其他异常: {e}")
                        print("Cannot Save svg !!!")
                print(f"Evaluating for {time.time() - t0} s")
                return
