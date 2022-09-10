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
        
        for batch in loader:
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
                
            if opts.train.debug:
                #########################
                # TODO: delete if nan issue is fixed
                if torch.isnan(loss):
                    # check architecture value
                    os.makedirs('debug', exist_ok=True)
                    import json
                    logger = {
                        'video': vid_enc_logger,
                        'spec': spec_enc_logger,
                        'audio': aud_enc_logger,
                        'projector': projector_logger,
                    }
                    with open(os.path.join('debug', 'weight_log.json'), 'w') as f:
                        json.dump(logger, f, indent=4)
                #########################

            # log status
            if opts.train.wandb:
                wandb.log({
                    # for report
                    'report/loss': loss.item(),
                    
                    # for debug
                    'debug/lr': optimizer.param_groups[0]['lr'],
                    'debug/loss_vs': criterion.modality_loss[('video', 'spec')]
                                        + criterion.modality_loss[('spec', 'video')],
                    'debug/loss_vw': criterion.modality_loss[('video', 'audio')]
                                        + criterion.modality_loss[('audio', 'video')],
                    'debug/loss_sw': criterion.modality_loss[('spec', 'audio')]
                                        + criterion.modality_loss[('audio', 'spec')],
                })

            # scheduler step
            with warmup_scheduler.dampening():
                scheduler.step()

            prog_bar.update()
            prog_bar.set_description(
                f"epoch: {round(step / len(loader), 3)} | "
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
    
    #########################
    # TODO: delete when nan issue is fixed
    # nan value debugging
    if opts.train.debug:
        def log_layer_output(logger, module_name):
            def hook(module, input, output):
                # logger[module_name] = output.detach()
                if type(output) is dict:
                    output = output['embedding']    # cnn14 res1dnet31
                output = output.detach()
                logger[module_name] = {
                    'abs_min': output.abs().min().item(),
                    'abs_max': output.abs().max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                }
            return hook
        
        vid_enc_logger = {}
        for name, module in model.vid_enc.named_modules():
            module.register_forward_hook(log_layer_output(vid_enc_logger, name))
        spec_enc_logger = {}
        for name, module in model.spec_enc.named_modules():
            module.register_forward_hook(log_layer_output(spec_enc_logger, name))
        aud_enc_logger = {}
        for name, module in model.aud_enc.named_modules():
            module.register_forward_hook(log_layer_output(aud_enc_logger, name))
        projector_logger = {}
        for name, module in model.projector.named_modules():
            module.register_forward_hook(log_layer_output(projector_logger, name))
    #########################
    



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