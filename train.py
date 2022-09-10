# imports

import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_warmup as warmup

from attrdict import AttrDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MMEncoder
from losses import MMConLoss
from dataset import MMDataset

from argparse import ArgumentParser


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='',
        help='Directory of dataset'
    )

    args = parser.parse_args()

    # load opts
    with open(args.config_path) as f:
        opts = AttrDict(yaml.safe_load(f))

    # determine device
    device = opts.train.device
    if not torch.cuda.is_available():
        assert 'cuda' not in opts.train.device, 'CUDA is not available.'


    # load model
    model = MMEncoder(opts)
    model.to(device)
    model.train()


    # load data
    train_dataset = MMDataset(opts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.train.batch_size,
        num_workers=opts.train.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )


    # define criterion
    criterion = MMConLoss(temperature=0.1)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=opts.train.lr)
    # define scheduler
    # https://github.com/Tony-Y/pytorch_warmup
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opts.train.max_step
    )
    warmup_scheduler = warmup.LinearWarmup(
        optimizer,
        warmup_period=opts.train.warmup_step
    )

    losses = []    # TODO: replace this with wandb

    for epoch in range(opts.train.epoch):
        prog_bar = tqdm(train_loader)
        for batch in prog_bar:
            # load batch to device
            vid_inp = batch['video'].to(device)
            spec_inp = batch['spec'].to(device)
            aud_inp = batch['audio'].to(device)

            optimizer.zero_grad()
            output = model(vid_inp, spec_inp, aud_inp)
            loss = criterion(output)
            loss.backward()
            optimizer.step()

            # status
            lr = optimizer.param_groups[0]['lr']
            loss_item = loss.item()
            losses.append(loss_item)

            with warmup_scheduler.dampening():
                lr_scheduler.step()

            prog_bar.update()
            prog_bar.set_description(f"epoch: {epoch} | loss: {round(loss_item, 3)}")

            
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()