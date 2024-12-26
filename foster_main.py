import sys

# sys.path.append("/share/home/kcli/CL_research/QuintCDKD")
sys.path.append("../ExemplarFreeCL")
from lib.approach.MI_DFCL import MI_DFCL_handler
from lib.dataset import *
from lib.config import foster_DFCL_cfg, update_config
from lib.utils.utils import (
    create_logger, split_classes_per_task,
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
        # default="./configs/foster_DFCL_cifar10.yaml",
        default="./configs/foster_DFCL_cifar100.yaml",
        # default="./configs/foster_DFCL_tiny.yaml",
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
    update_config(foster_DFCL_cfg, args)
    logger, log_file = create_logger(foster_DFCL_cfg, "log")
    warnings.filterwarnings("ignore")
    # order_class_list = [52, 1, 30, 96, 19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 60, 53, 95, 74, 92, 26, 59, 46, 90, 70,
    #                     50,
    #                     44, 76, 55, 21, 61, 6, 63, 42, 34, 84, 35, 39, 45, 4, 5, 48, 32, 20, 83, 58, 47, 80, 17, 67,
    #                     81, 7, 87, 97, 98, 99, 24, 10, 86, 56, 71, 23, 22, 91, 94, 18, 27, 88, 57, 31, 65, 12, 82,
    #                     75, 25, 13, 69, 77, 85, 51, 49, 78, 72, 33, 62, 54, 11, 16, 36, 40, 0, 73, 8, 29, 93, 89,
    #                     28, 64, 15, 9]
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    split_seleted_data = None
    # split_seleted_data = split_classes_per_task(order_class_list, tasks=foster_DFCL_cfg.DATASET.all_tasks)
    dataset_split_handler = eval(foster_DFCL_cfg.DATASET.dataset)(foster_DFCL_cfg, split_selected_data=split_seleted_data)
    if foster_DFCL_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = foster_DFCL_cfg.availabel_cudas
        device_ids = [i for i in range(len(foster_DFCL_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    midfcl_handler = MI_DFCL_handler(dataset_split_handler, foster_DFCL_cfg, logger)
    midfcl_handler.midfcl_train_main()
