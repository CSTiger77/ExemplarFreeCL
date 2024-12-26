import sys

# sys.path.append("/share/home/kcli/CL_research/QuintCDKD")
sys.path.append("../ExemplarFreeCL")
from lib.approach.ZeroShotKD import ZeroShotKD_handler
from lib.dataset import *
from lib.config import ZeroShotKD_cfg, update_config
from lib.utils.utils import (
    create_logger, split_classes_per_task,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for ABD")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="./configs/ZeroShotKD_cifar10.yaml",
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
    update_config(ZeroShotKD_cfg, args)
    logger, log_file = create_logger(ZeroShotKD_cfg, "log")
    warnings.filterwarnings("ignore")
    split_seleted_data = None
    dataset_split_handler = eval(ZeroShotKD_cfg.DATASET.dataset)(ZeroShotKD_cfg, split_selected_data=split_seleted_data)
    if ZeroShotKD_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = ZeroShotKD_cfg.availabel_cudas
        device_ids = [i for i in range(len(ZeroShotKD_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    zeroShotKD_handler = ZeroShotKD_handler(dataset_split_handler, ZeroShotKD_cfg, logger)
    # zeroShotKD_handler.deepInvert_KD_train_main()
    zeroShotKD_handler.GAN_KD_train_main()
