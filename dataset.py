import os
import sys
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_dataloader(dset_dir, batch_size, is_training, img_size, dset_name="MNIST"):
    assert batch_size > 1
    dset_cls = getattr(datasets, dset_name)
    dset = dset_cls(
            dset_dir,
            train=is_training,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
    dataloader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=is_training,
    )
    logger.info(
        "Loading {} dataset from directory: {}, "
        "batch_size: {}, "
        "img_size: {}, "
        "is_training: {}.".format(
            dset_name, dset_dir, batch_size, img_size, is_training
        )
    )
    return dataloader