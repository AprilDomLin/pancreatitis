# coding: utf-8
"""
For training and evaluating the models including ViT and ResNet.

----------------------------------
Version    : 0.0.1
Date       : 2023/10/18   16:39
----------------------------------
Author     : April
Contact    : fanglwh@foxmail.com
"""
import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddleclas
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd
from PIL import Image

# Model
class PancreatitisNet(paddle.nn.Layer):
    def __init__(self,
                 model_name: str,
                 class_num: int = 2,
                 img_size: int = 224,
                 pretrained: bool = True,
                 ) -> None:
        super(PancreatitisNet, self).__init__()
        self.model_name = model_name
        self.class_num = class_num
        self.img_size = img_size
        self.pretrained = pretrained

        # Initialize the model and Load pretrained paramters
        self.base = getattr(paddleclas, model_name)(pretrained=pretrained, class_num=class_num)

    def forward(self, x):
        x = self.base(x)  # Base output from pretrained model
        return x


# DataLoader
class CTImage(paddle.io.Dataset):
    def __init__(self,
                 sample_path: str,
                 shape: int = 224,
                 ) -> None:
        super(CTImage, self).__init__()
        self.data = pd.read_csv(sample_path)[:]
        self.shape = (shape, shape)

    def __getitem__(self, idx: int) -> dict:
        path, category = self.data.loc[idx, 'path'], self.data.loc[idx, 'cls']
        # print(path)
        img = paddle.vision.image_load(path).convert("RGB")  # Load image as 512 * 512 * 3
        img = img.resize(self.shape, Image.ANTIALIAS)  # Resize shape into 224 * 224 * 3
        img = np.array(img).astype("float32").transpose(2, 0, 1)  # Transpose dims as 3 * 224 * 224
        img = img / 255.0  # Normalize
        return dict(input=img, target=int(category), path=path)

    def __len__(self) -> int:
        return len(self.data)


def main():
    # Train & Evaluate
    dataset_train = paddle.io.DataLoader(
        CTImage(sample_path=DATAPATH_TRAIN, shape=config.img_size),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    dataset_eval = paddle.io.DataLoader(
        CTImage(sample_path=DATAPATH_EVALUATE, shape=config.img_size),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    model = PancreatitisNet(model_name=config.model_name,
                            class_num=2,
                            img_size=config.img_size,
                            pretrained=True
                            )

    para = [p for p in model.base.parameters()]
    # parameters of forecast layer
    optimizer_fc = paddle.optimizer.Adam(parameters=para[:-2], learning_rate=config.lr_fc)

    # parameters of base layer
    optimizer_bs = paddle.optimizer.Adam(parameters=para[-2:], learning_rate=config.lr_bs)

    criterion = paddle.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        print('-' * 20, f'Epoch: {epoch:02d}', '-' * 20)
        if (epoch % 2 == 0) & (epoch != 0):
            optimizer_fc.set_lr(optimizer_fc.get_lr() * 0.90)
        if (epoch % 2 == 0) & (epoch != 0):
            optimizer_bs.set_lr(optimizer_bs.get_lr() * 0.75)

        loss_all = 0
        model.train()
        for batch_id, data in enumerate(dataset_train):
            x_data = paddle.to_tensor(data['input'], "float32")
            y_data = paddle.to_tensor(data['target'], "int64")
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss.backward()
            optimizer_fc.minimize(loss)
            optimizer_bs.minimize(loss)
            loss_all += loss.item()
            model.clear_gradients()
        loss_all = float(loss_all) / dataset_train.__len__()
        save_path = Path(f'./checkpoint/{config.model_name}/{epoch:02d}.pdparams')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        paddle.save(model.state_dict(), str(save_path))

        model.eval()
        acc, f1 = [], []
        with paddle.no_grad():
            for batch_id, data in enumerate(dataset_eval):
                x_data = paddle.to_tensor(data['input'], "float32")
                y_pred = model(x_data)

                y_data_np = np.array(data['target'])
                y_pred_np = np.argmax(y_pred.numpy(), axis=1)

                tp = np.sum((y_data_np == 1) & (y_pred_np == 1))
                fp = np.sum((y_data_np == 0) & (y_pred_np == 1))
                fn = np.sum((y_data_np == 1) & (y_pred_np == 0))
                tn = np.sum((y_data_np == 0) & (y_pred_np == 0))

                tmp_acc = (tp + tn) / (tp + fp + fn + tn)
                tmp_f1 = 2 * tp / (2 * tp + fp + fn)
                acc.append(tmp_acc)
                f1.append(tmp_f1)
        print(f">>>Model Name: {config.model_name}\t Image Size: {config.img_size}\n"
              f">>>Loss: {loss_all:.6f}\n"
              f">>>Accuracy: {np.array(acc).mean():.4f}\n"
              f">>>F1-Score: {np.array(f1).mean():.4f}"
              )


if __name__ == '__main__':

    random.seed(1234)
    np.random.seed(1234)
    paddle.seed(1234)

    # configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='ViT_large_patch16_224', help='The name of model in paddle')
    parser.add_argument('--img_size', type=int, default=224, help='The size of input image')
    parser.add_argument('--class_num', type=int, default=2, help='The numbers to be classified')

    parser.add_argument('--epochs', type=int, default=20, help='The epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training and evaluation')
    parser.add_argument('--lr_fc', type=float, default=4e-4, help='Learning Rate of forecast layer')
    parser.add_argument('--lr_bs', type=float, default=5e-5, help='Learning Rate of basic layers')

    config = parser.parse_args()

    DATAPATH_TRAIN = "./src/train_list.csv"
    DATAPATH_EVALUATE = "./src/eval_list.csv"

    main()
