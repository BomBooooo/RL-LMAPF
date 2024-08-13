import torch
import datetime
import pathlib

from tools import save_json, set_seed_everywhere
from Trainer import Trainer
from Evaluater import Evaluater
from paraser import get_args_parser


def main(args):
    date = datetime.datetime.now().strftime("%m%d.%H%M.%S")
    map_file = args.map_file.split("/")[1].split(".")[0]
    if args.isEval:
        model_name = args.isLoad.split("/")[1].split(".")[0]
        args.logdir = (
            pathlib.Path("eval_logs").expanduser().absolute()
            / f"{date}_{map_file}_{model_name}_{args.info}"
        )
    else:
        args.logdir = (
            pathlib.Path("logs").expanduser().absolute()
            / f"{date}_{map_file}_{args.isLoad is not None}_{args.agent_num}_{args.lr}_{args.batch_size}_{args.info}"
        )
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.evaldir = args.logdir / "eval"
    args.evaldir.mkdir(parents=True, exist_ok=True)
    if args.isRender:
        args.svgdir = args.logdir / "svg"
        args.svgdir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.logdir / "args.json")
    args.device = torch.device(
        f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    )
    args.buffer_size = min(args.agent_num * args.time_limit * 2, 1e6)
    set_seed_everywhere(args.seed)

    if args.isEval:
        evaluater = Evaluater(args)
        evaluater()
    else:
        print(" " * 26 + "Options")
        for k, v in vars(args).items():
            print(" " * 26 + k + ": " + str(v))
        print("Logdir", args.logdir)
        args.modeldir = args.logdir / "model"
        args.modeldir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(args)
        trainer()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
