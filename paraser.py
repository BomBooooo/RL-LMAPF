import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Dreamer training and evaluation script for Fulfillment system environment",
        add_help=False,
    )
    parser.add_argument("--info", default="train")
    parser.add_argument("--debug", default=False)
    parser.add_argument("--agent_name", default="RL-LMAPF")
    # eval
    # parser.add_argument("--isEval", default=True, type=bool)
    # parser.add_argument("--isLoad", default="model/Berlin_512.pth")
    # parser.add_argument("--map_file", default="mapf-map-porcess/Berlin_1_256.map")
    # train
    parser.add_argument("--isEval", default=False, type=bool)
    parser.add_argument("--isLoad", default=None)
    parser.add_argument("--map_file", default="mapf-map-porcess/Berlin_1_256.map")
    # environment
    parser.add_argument("--env_name", default="GridMap")
    parser.add_argument("--time_limit", default=500)
    parser.add_argument("--obs_radius", default=5, type=int)
    parser.add_argument("--nbr_radius", default=5, type=int)
    parser.add_argument("--obs_dim", default=3, type=int)
    parser.add_argument("--action_dim", default=5, type=int)
    parser.add_argument("--agent_num", default=16, type=int)
    # render
    parser.add_argument("--isRender", default=True, type=bool)
    parser.add_argument("--time_interval", default=0.4)
    parser.add_argument("--render_cell_size", default=15)
    # defaults
    parser.add_argument("--seed", default=2024)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--modeldir", default=None)
    parser.add_argument("--traindir", default=None)
    parser.add_argument("--evaldir", default=None)
    parser.add_argument("--svgdir", default=None)
    parser.add_argument("--steps", default=100000)
    parser.add_argument("--log_every", default=100)
    parser.add_argument("--eval_every", default=2500)
    parser.add_argument("--eval_episode_num", default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_id", default="0")  #
    parser.add_argument("--dtype", default=torch.int32)
    # Training
    parser.add_argument("--batch_size", default=2048)
    parser.add_argument("--hidden_dim", default=256)
    parser.add_argument("--msg_dim", default=512)
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--norm", default="none")
    parser.add_argument("--dropout_p", default=0)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--eps", default=1e-5)
    parser.add_argument("--wd", default=1e-6)
    parser.add_argument("--centralized", default=None)
    parser.add_argument("--update_buffer", default=1000)
    parser.add_argument("--buffer_size", default=100000)
    parser.add_argument("--opt", default="adam")
    # RL
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--epsilon", default=(0.9, 0.1))
    parser.add_argument("--target_update", default=50)

    return parser
