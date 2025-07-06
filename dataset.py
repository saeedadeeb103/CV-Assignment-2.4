from os.path import isdir, join
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10


CIFAR_IDX_TO_LABEL = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

class CIFAR10Dataset:
    X_mean: torch.Tensor
    X_std: torch.Tensor
    data_shape: torch.Size

    def __init__(self, data_dir: str = "data"):
        download = not isdir(join(data_dir, "cifar-10-batches-py"))
        self.train_dataset = CIFAR10(root="data", download=download, train=True)
        self.test_dataset = CIFAR10(root="data", train=False)

    def get_splits(
        self,
        device: str,
        trainval_split: float,
        preprocess: bool = True,
        flatten: bool = True,
        include_bias: bool = False,
        X_dtype: torch.dtype = torch.float32,
    ) -> tuple:
        X_train = (
            torch.tensor(self.train_dataset.data)
            .permute(0, 3, 1, 2)
            .to(X_dtype)
            .div_(255)
        )
        y_train = torch.tensor(self.train_dataset.targets)
        X_test = (
            torch.tensor(self.test_dataset.data)
            .permute(0, 3, 1, 2)
            .to(X_dtype)
            .div_(255)
        )
        y_test = torch.tensor(self.test_dataset.targets)

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        if preprocess:
            X_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
            X_std = X_train.std(dim=(0, 2, 3), keepdim=True)
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            self.X_mean = X_mean[0]
            self.X_std = X_std[0]

        if flatten:
            self.data_shape = X_train.shape[-2:]
            X_train = X_train.view(len(X_train), -1)
            X_test = X_test.view(len(X_test), -1)

        if include_bias:
            X_train = torch.cat(
                [X_train, torch.ones(len(X_train), 1, device=device)], dim=1
            )
            X_test = torch.cat(
                [X_test, torch.ones(len(X_test), 1, device=device)], dim=1
            )

        n_train = int(len(X_train) * trainval_split)
        X_val = X_train[n_train:]
        y_val = y_train[n_train:]

        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def visualize_samples(self, X: torch.Tensor, y: torch.Tensor, to_show: int = 4):
        # unnormalize image
        samples = []
        for y_i, cate in enumerate(CIFAR_IDX_TO_LABEL):
            plt.text(-4, 34 * y_i + 18, cate, ha="right")
            (idxs,) = (y == y_i).nonzero(as_tuple=True)
            for i in range(to_show):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                samples.append(self._unnormalize(X[idx]))
        img = torchvision.utils.make_grid(samples, nrow=to_show)

        plt.imshow(self._tensor_to_img(img))
        plt.axis("off")
        plt.show()

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 2:
            if x.shape[-1] % 2 != 0:
                x = x[:-1]
            x = x.reshape((-1, *self.data_shape))
            if self.X_std is not None and self.X_mean is not None:
                x = (x * self.X_std) + self.X_mean
        return x

    def _tensor_to_img(self, x: torch.Tensor) -> np.ndarray:
        x = x.permute(1, 2, 0) * 255
        ndarr = x.cpu().to(torch.uint8).numpy()
        return ndarr
