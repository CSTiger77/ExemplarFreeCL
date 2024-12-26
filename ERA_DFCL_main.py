import sys

sys.path.append("../ExemplarFreeCL")
from lib.approach.MI_DFCL import MI_DFCL_handler
from lib.dataset import *
from lib.config import ERA_DFCL_cfg, update_config
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
        # default="./configs/ERA_DFCL_cifar10.yaml",
        default="./configs/ERA_DFCL_cifar100.yaml",
        # default="./configs/ERA_DFCL_tiny.yaml",
        # default="./configs/ERA_DFCL_imagenet.yaml",
        # default="./configs/ERA_DFCL_for_CAM_imagenet.yaml",
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
    update_config(ERA_DFCL_cfg, args)
    logger, log_file = create_logger(ERA_DFCL_cfg, "log")
    warnings.filterwarnings("ignore")
    order_class_list = [52, 1, 30, 96, 19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 60, 53, 95, 74, 92, 26, 59, 46, 90, 70, 50,
                        44,76,55, 21, 61, 6, 63, 42, 34, 84, 35, 39, 45, 4, 5, 48, 32, 20, 83, 58, 47, 80, 17, 67,
                        81,7,87, 97, 98, 99, 24, 10, 86, 56, 71, 23, 22, 91, 94, 18, 27, 88, 57, 31, 65, 12, 82,
                        75,25, 13, 69, 77, 85, 51, 49, 78, 72, 33, 62, 54, 11, 16, 36, 40, 0, 73, 8, 29, 93, 89,
                        28,64,15, 9]
    # if ERA_DFCL_cfg.seed == 10:
    #     if "CIFAR100" == ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60, 53, 95, 74, 92, 26, 59, 46, 90, 70, 50,
    #                             44,
    #                             76,
    #                             55, 21, 61, 6, 63, 42, 34, 84, 52, 35, 39, 45, 4, 5, 48, 32, 20, 83, 58, 47, 80, 17, 67,
    #                             81,
    #                             7,
    #                             87, 97, 98, 99, 24, 10, 96, 86, 56, 71, 23, 22, 91, 94, 18, 27, 88, 57, 31, 65, 12, 82,
    #                             30,
    #                             75,
    #                             25, 13, 69, 77, 85, 51, 49, 78, 72, 33, 62, 54, 11, 16, 36, 40, 0, 73, 8, 29, 93, 89,
    #                             28,
    #                             64,
    #                             15, 9]
    #     elif "CIFAR10" == ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [8, 2, 5, 6, 3, 1, 0, 7, 4, 9]
    #     elif "tiny" in ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [59, 5, 20, 198, 52, 19, 162, 55, 69, 2, 98, 10, 75, 142, 124, 63, 109, 78, 111, 185,
    #                             154, 130, 61, 87, 102, 121, 136, 1, 47, 172, 159, 39, 76, 91, 35, 178, 127, 169, 46,
    #                             174, 190, 7, 26, 138, 58, 72, 103, 199, 56, 116, 24, 43, 101, 163, 21, 60, 175, 70, 90,
    #                             49, 119, 110, 95, 167, 193, 68, 165, 114, 67, 66, 120, 38, 196, 161, 99, 152, 83, 166,
    #                             117, 41, 80, 81, 32, 170, 48, 25, 53, 105, 17, 194, 51, 14, 82, 84, 184, 29, 3, 23, 147,
    #                             188, 37, 189, 186, 187, 45, 132, 97, 179, 191, 42, 129, 131, 79, 160, 177, 143, 168, 12,
    #                             112, 11, 22, 106, 85, 146, 6, 128, 149, 155, 148, 104, 34, 108, 50, 134, 144, 145, 4,
    #                             133, 44, 96, 176, 28, 135, 171, 180, 71, 118, 192, 197, 137, 74, 182, 94, 151, 150, 173,
    #                             93, 18, 27, 36, 57, 31, 65, 140, 89, 158, 30, 86, 92, 141, 126, 153, 13, 77, 181, 183,
    #                             33, 62, 122, 107, 88, 54, 139, 100, 16, 115, 164, 40, 0, 73, 8, 195, 157, 156, 123, 113,
    #                             64, 15, 125, 9]
    #     else:
    #         raise ValueError(f"ERA_DFCL_cfg.DATASET.dataset_name")
    # elif ERA_DFCL_cfg.seed == 100:
    #     if "CIFAR100" == ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [37, 62, 26, 41, 35, 25, 36, 33, 77, 21, 85, 50, 92, 69, 96, 78, 72, 5, 40, 11, 29, 83,
    #                             82,
    #                             43, 28, 22, 23, 90, 86, 20, 32, 6, 3, 12, 51, 84, 73, 64, 54, 68, 75, 74, 57, 42, 76,
    #                             99,
    #                             17, 93, 63, 0, 18, 44, 38, 45, 39, 70, 94, 30, 71, 46, 56, 80, 91, 88, 19, 81, 55, 89,
    #                             61,
    #                             65, 47, 49, 7, 97, 59, 95, 13, 1, 31, 4, 27, 2, 9, 16, 58, 60, 15, 98, 34, 14, 66, 53,
    #                             52,
    #                             10, 48, 79, 87, 67, 24, 8]
    #     elif "CIFAR10" == ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [7, 6, 1, 5, 4, 2, 0, 3, 9, 8]
    #     elif "tiny" in ERA_DFCL_cfg.DATASET.dataset_name:
    #         order_class_list = [126, 104, 99, 92, 111, 167, 116, 96, 52, 69, 164, 124, 182, 154, 125, 196, 194, 177,
    #                             163, 31, 11, 73, 15, 41, 97, 128, 133, 82, 139, 123, 83, 65, 151, 162, 170, 77, 32, 173,
    #                             174, 85, 168, 112, 171, 181, 7, 46, 75, 28, 29, 195, 40, 153, 115, 64, 59, 1, 192, 136,
    #                             152, 161, 74, 3, 185, 26, 90, 127, 81, 88, 119, 110, 57, 44, 148, 160, 89, 146, 199, 10,
    #                             20, 165, 12, 16, 101, 120, 45, 142, 117, 184, 187, 183, 51, 39, 118, 37, 6, 54, 25, 21,
    #                             48, 9, 23, 35, 175, 50, 62, 140, 169, 19, 122, 55, 178, 22, 158, 102, 190, 33, 76, 150,
    #                             114, 95, 84, 42, 189, 191, 193, 149, 71, 5, 36, 43, 157, 172, 134, 70, 131, 179, 145, 0,
    #                             78, 166, 68, 188, 156, 30, 106, 13, 72, 17, 18, 38, 109, 47, 113, 56, 27, 63, 147, 105,
    #                             121, 2, 80, 186, 61, 49, 135, 197, 91, 4, 100, 141, 129, 159, 132, 108, 155, 130, 86,
    #                             93, 137, 144, 58, 60, 107, 143, 198, 34, 14, 66, 53, 98, 180, 94, 138, 176, 79, 87, 103,
    #                             67, 24, 8]
    #     else:
    #         raise ValueError(f"ERA_DFCL_cfg.DATASET.dataset_name")
    # elif ERA_DFCL_cfg.seed == 0:
    #     order_class_list = None
    # else:
    #     order_class_list = None
    # else:
    #     raise ValueError(f"ERA_DFCL_cfg.seed == 0, 10, 100")
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    # split_seleted_data = split_classes_per_task(order_class_list, tasks=ERA_DFCL_cfg.DATASET.all_tasks)
    split_seleted_data = None
    dataset_split_handler = eval(ERA_DFCL_cfg.DATASET.dataset)(ERA_DFCL_cfg, split_selected_data=split_seleted_data)
    if ERA_DFCL_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = ERA_DFCL_cfg.availabel_cudas
        device_ids = [i for i in range(len(ERA_DFCL_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    midfcl_handler = MI_DFCL_handler(dataset_split_handler, ERA_DFCL_cfg, logger)

    if "imagenet" in ERA_DFCL_cfg.DATASET.dataset_name and \
            "tiny" not in ERA_DFCL_cfg.DATASET.dataset_name:
        midfcl_handler.midfcl_train_main_for_local_dataset()
    else:
        midfcl_handler.midfcl_train_main()
