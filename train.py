# imports

import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import pytorch_warmup as warmup

from attrdict import AttrDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MMEncoder
from losses import MMConLoss
from dataset import MMDataset

from argparse import ArgumentParser


def train(
    opts,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    warmup_scheduler,
):
    step = 0
    max_epoch = -(-opts.train.max_step // len(loader))
    prog_bar = tqdm(range(opts.train.max_step))
    
    if opts.train.fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(max_epoch):
        
        for batch in train_loader:
            if step == opts.train.max_step:
                break
            step += 1
            
            # load batch to device
            vid_inp = batch['video'].to(device)
            spec_inp = batch['spec'].to(device)
            aud_inp = batch['audio'].to(device)

            optimizer.zero_grad()
            
            if opts.train.fp16:
                with torch.cuda.amp.autocast():
                    output = model(vid_inp, spec_inp, aud_inp)
                    loss = criterion(output)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                output = model(vid_inp, spec_inp, aud_inp)
                loss = criterion(output)
                loss.backward()
                optimizer.step()
                

            # status
            if opts.train.wandb:
                wandb.log({
                    'lr': optimizer.param_groups[0]['lr'],
                    'loss': loss.item(),
                    # 'loss_vs': ,
                    # 'loss_vw': ,
                    # 'loss_sw': ,
                })

            # scheduler step
            with warmup_scheduler.dampening():
                lr_scheduler.step()

            prog_bar.update()
            prog_bar.set_description(
                f"epoch: {round(step / len(loader), 3)} | ",
                f"loss: {round(loss.item(), 3)}"
            )




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

    if opts.train.wandb:
        
        # parameter validation
        assert 'tmp' not in opts.train.wandb_exp_name, "Set wandb_exp_name"
        # login
        wandb.login()

        # init_wandb
        wandb.init(
            project="multi-modal-embedding", 
            name=opts.train.wandb_exp_name, 
            config=opts,
        )

    # determine device
    device = opts.train.device
    if not torch.cuda.is_available():
        assert 'cuda' not in opts.train.device, "CUDA is not available."


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


    train(opts, model, train_loader, criterion, 
          optimizer, lr_scheduler, warmup_scheduler)
    
    if opts.train.wandb:
        wandb.finish()