import sys
sys.path.append("../ExemplarFreeCL")
from lib.approach.contest_DFCL import contest_handler
from lib.config import contest_cfg, update_config
from lib.utils.utils import (
    create_logger,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for contest_DFCL")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="./configs/contest_10splitTasks.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(contest_cfg, args)
    logger, log_file = create_logger(contest_cfg, "log")
    warnings.filterwarnings("ignore")
    if contest_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = contest_cfg.availabel_cudas
        device_ids = [i for i in range(len(contest_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    midfcl_handler = contest_handler(contest_cfg, logger)
    midfcl_handler.contest_train_main()
