import sys

sys.path.append("../ExemplarFreeCL")
from lib.approach.NAPAVQ import NAPAVQ_handler
from lib.dataset import *
from lib.config import NAPAVQ_cfg, update_config
from lib.utils.utils import (
    create_logger,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for EARS-DFCL")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        # default="./configs/NAPAVQ_cifar10.yaml",
        default="./configs/NAPAVQ_cifar100.yaml",
        # default="./configs/NAPAVQ_tiny.yaml",
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
    update_config(NAPAVQ_cfg, args)
    logger, log_file = create_logger(NAPAVQ_cfg, "log")
    warnings.filterwarnings("ignore")
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    split_seleted_data = None
    dataset_split_handler = eval(NAPAVQ_cfg.DATASET.dataset)(NAPAVQ_cfg, split_selected_data=split_seleted_data)
    if NAPAVQ_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = NAPAVQ_cfg.availabel_cudas
        device_ids = [i for i in range(len(NAPAVQ_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    napavq_handler = NAPAVQ_handler(dataset_split_handler, NAPAVQ_cfg, logger)
    napavq_handler.napavq_train_main()
