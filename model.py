import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(nn.Module):
    def __init__(self, idim: int, img_size: tuple):
        """
        References: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

        Args:
            idim: the hidden dim of generator. It should be the same as the `args.input_dim` in `train.py`
            img_size: the size of expected images
        """
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.idim = idim
        self.odim = np.prod(img_size)
        self.img_size = img_size

        self.input_emb = block(idim, 128, normalize=False)
        self.convs = nn.ModuleList([
            block(128, 256),
            block(256, 512),
            block(512, 1024),
        ])
        self.lin = nn.Linear(1024, self.odim)
        self.tanh = nn.Tanh()

        logger.info(self)

    def forward(self, x):
        out = self.input_emb(x)
        for conv in self.convs:
            out = conv(out)
        out = self.tanh(self.lin(out))
        out = out.view(out.size(0), *self.img_size)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size):
        """
        References: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

        Args:
            img_size: the size of input images
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.idim = np.prod(img_size)

        self.model = nn.Sequential(
            nn.Linear(self.idim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        logger.info(self)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        p = self.model(x)
        return p