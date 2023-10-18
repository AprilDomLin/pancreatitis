# coding: utf-8
"""
Interpret the ViT models

----------------------------------
Version    : 0.0.1
Date       : 2023/10/18   18:58
----------------------------------
Author     : April
Contact    : fanglwh@foxmail.com
"""

import argparse
from pathlib import Path
import random

import interpretdl as it
import numpy as np
import paddle
import pandas as pd

from main import PancreatitisNet


def main():
    # dataset paths
    dataset_eval = pd.read_csv(DATAPATH_EVALUATE)

    # Reload model parameters
    model = PancreatitisNet(model_name=config.model_name,
                            class_num=config.class_num,
                            img_size=config.img_size,
                            pretrained=False,
                            )
    checkpoint =  Path(f'./checkpoint/{config.model_name}/{config.epoch:02d}.pdparams')
    if not checkpoint.exists():
        raise FileNotFoundError(f'The checkpoint is not exists, please check it: {checkpoint.absolute()}')

    model.set_state_dict(paddle.load(str(checkpoint.absolute())))

    save_dir = Path(config.save_dir).joinpath(f'GAI_{config.epoch:02d}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # GAI
    ga = it.GACVInterpreter(model, device='gpu:0')
    for col in dataset_eval.itertuples():
        save_path = save_dir.joinpath(Path(col.path).name)
        ga.interpret(
            str(col.path),
            start_layer=config.start_layer,
            resize_to=config.img_size,
            label=int(col.cls),
            visual=False,
            attn_map_name=config.attn_map_name,
            save_path=save_path)


if __name__ == '__main__':

    random.seed(1234)
    np.random.seed(1234)
    paddle.seed(1234)

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./interpret', help='The folder to save the result')

    parser.add_argument('--model_name', type=str, default='ViT_large_patch16_224', help='The name of model in paddle')
    parser.add_argument('--img_size', type=int, default=224, help='The size of input image')
    parser.add_argument('--class_num', type=int, default=2, help='The numbers to be classified')

    parser.add_argument('--epoch', type=int, default=4, help='The epoch id to interpret')

    parser.add_argument('--start_layer', type=int, default=6, help='')
    parser.add_argument('--attn_map_name', type=str, default='^base.blocks.[0-9]*.attn.attn_drop$', help='')

    config = parser.parse_args()

    DATAPATH_EVALUATE = "./src/eval_list.csv"

    main()