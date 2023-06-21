import os
import sys
import time
import argparse
import logging

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import tensorboardX



from model import Generator, Discriminator
from dataset import get_dataloader
from utils import AverageMeter

# References: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_tag", type=str, help="The tag of current experiment")
    parser.add_argument("--dset_dir", type=str, default="./data/mnist", help="where to load mnist dataset")
    parser.add_argument("--save_dir", type=str, default="./generated", help="where to save generated images")
    parser.add_argument("--log_dir", type=str, default="./.checkpoints", help="where to save tensorboard logs")
    parser.add_argument("--img_size", type=int, default=(1, 28, 28), help="size of each image dimension")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")

    parser.add_argument(
        "--update_g_per_iter", default=1, type=int, help="How many updates the generator performs in each iteration"
    )
    parser.add_argument(
        "--update_d_per_iter", default=1, type=int, help="How many updates the discriminator performs in each iteration"
    )
    # Notes: Typically, we don't stop updating the discriminator during training. Here we only add this option for
    # demonstration purpose.
    parser.add_argument(
        "--d_stop_update", default=int(1e10), type=int,
        help="Which epoch the discriminator stops to update"
    )

    parser.add_argument("--loss_d_scale", type=float, default=2., help="The scaling factor of the discriminator's loss")
    parser.add_argument("--input_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    parser.add_argument("--use_cuda", action="store_true", help="Whether to use CUDA")

    args = parser.parse_args()
    for k in ("save_dir", "log_dir"):
        v = getattr(args, k)
        v = os.path.join(v, args.exp_tag)
        setattr(args, k, v)
        os.makedirs(os.path.expanduser(v), exist_ok=True)
    return args


if __name__ == "__main__":
    # Notes: For reproducibility, we often fix the random seeds(e.g. torch, numpy, random) in the very beginning of training
    # Here we omit this step since we don't need any guarantee of reproducibility.
    # References: https://pytorch.org/docs/stable/notes/randomness.html
    args = get_args()
    logger.info(f"ARGS: {args}")
    if torch.cuda.is_available() and args.use_cuda:
        cuda = True
        logger.info("Using CUDA")
    else:
        cuda = False
        logger.info("Using CPU")

    # Define model, loss function(s), dataloader, optimizers and all other stuffs
    G = Generator(args.input_dim, args.img_size)
    D = Discriminator(args.img_size)
    ce = torch.nn.BCELoss()
    # In each iteration, we may update the discriminator for multiple times, so we directly load `update_d_per_iter`
    # batchs into memory, so we set the `batch_size` as `args.batch_size * args.update_d_per_iter` instead of
    # `args.batch_size`
    dataloader = get_dataloader(
        args.dset_dir, args.batch_size * args.update_d_per_iter,
        is_training=True, img_size=args.img_size[1:]
    )
    if cuda:
        # Notes: if you have more than one GPUs, you can use DataParallel(DP) or DistributedDataParallel(DDP) to enable
        # parallelness among multiple devices. DP is easier to implement but may be slower than DDP.
        # In this script, we only utilize one GPU so we don't need any of them.
        # One thing to notice is that DDP uses multiple processes while DP uses only one process.
        # References:
        #   https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
        #   https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        G.cuda()
        D.cuda()
        ce.cuda()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    d = os.path.join(args.log_dir, "log")
    os.makedirs(d, exist_ok=True)
    writer = tensorboardX.SummaryWriter(log_dir=d)

    # Start training
    tic = time.time()
    n_iter = 0
    # p_fake_is_real: the probability that the discriminator thinks the fake images are real.
    recorders = {
        k: AverageMeter() for k in ('d', 'g', "p_real_is_real", "p_fake_is_real", "p_fake_is_fake")
    }
    d_stoped = False
    for epoch_idx in range(1, args.n_epochs + 1):   # indexing from 1 instead of 0
        if epoch_idx >= args.d_stop_update:
            logger.info(f"Epoch {epoch_idx}: Stop updating the discriminator.")
            d_stoped = True
            for n, p in D.named_parameters():
                p.requires_grad_(False)
        pbar = tqdm.tqdm(range(len(dataloader)), desc="Training", disable=False)
        pbar.set_postfix({"epoch": f"{0}/{args.n_epochs}", "loss_d": 0., "loss_g": 0.})

        def d_step(img):
            # Perform one step of discriminator
            loss_real_is_real = ce(D(img), real)
            loss_fake_is_fake = ce(D(fake_imgs.detach()), fake)
            loss_d = (loss_real_is_real + loss_fake_is_fake) / args.loss_d_scale
            return loss_d, loss_real_is_real, loss_fake_is_fake

        for i, (imgs, _) in enumerate(dataloader):
            B = imgs.size(0) // args.update_d_per_iter
            n_iter += 1

            # Prepare labels for adversarial training
            real = torch.ones((B, 1), requires_grad=False)
            fake = torch.zeros((B, 1), requires_grad=False)
            if cuda:
                imgs, real, fake = [t.cuda() for t in (imgs, real, fake)]

            for j in range(args.update_g_per_iter):
                #  Train generator
                noise = torch.randn((B, args.input_dim))
                if cuda:
                    noise = noise.cuda()
                fake_imgs = G(noise)
                loss_fake_is_real = ce(D(fake_imgs), real)
                loss_g = loss_fake_is_real
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()
                recorders['g'].update(loss_g.item(), 1)
                recorders["p_fake_is_real"].update(np.exp(-loss_fake_is_real.item()), 1)

            assert imgs.size(0) == args.batch_size * args.update_d_per_iter
            for j in range(args.update_d_per_iter):
                #  Train discriminator
                img = imgs[j * args.batch_size: (j + 1) * args.batch_size]
                if not d_stoped:
                    loss_d, loss_real_is_real, loss_fake_is_fake = d_step(img)
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()
                else:
                    with torch.no_grad():
                        loss_d, loss_real_is_real, loss_fake_is_fake = d_step(img)
                recorders['d'].update(loss_d.item(), 1)
                recorders["p_real_is_real"].update(np.exp(-loss_real_is_real.item()), 1)
                recorders["p_fake_is_fake"].update(np.exp(-loss_fake_is_fake.item()), 1)

            state = {
                "epoch": f"{epoch_idx}/{args.n_epochs}",
                "loss_d": f"{recorders['d'].get():.4f}",
                "loss_g": f"{recorders['g'].get():.4f}",
            }
            pbar.set_postfix(state)
            pbar.update()
            writer.add_scalars("Training", {k: v.get() for k, v in recorders.items()}, n_iter)

            # Save images per `args.sample_interval` iterations
            batches_done = epoch_idx * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(fake_imgs.data[:25], f"{args.save_dir}/{batches_done:07d}.png", nrow=5, normalize=True)
        # pbar.reset()

        ckpt_ph = os.path.join(
            args.log_dir,
            "train", f"epoch{epoch_idx}_lossd_{recorders['d'].get():.3f}_lossg_{recorders['g'].get():.3f}.pt"
        )
        os.makedirs(os.path.dirname(ckpt_ph), exist_ok=True)
        # Notes: if you're using DataParallel or DistributedDataParallel, you may prefer G.module.state_dict() and
        # D.module.state_dict() to unwrap G and D first.
        torch.save(
            {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
            },
            ckpt_ph
        )

    logger.info(f"Training finished. Duration: {(time.time() - tic) / 3600:.2f}h")
