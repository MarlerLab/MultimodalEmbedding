# import 
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from attrdict import AttrDict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MMEncoder
from dataset import MMDataset



# define function where you put modality and get print out the plot
def analyze_embedding(modality, samples, reduction_model):
    num_labels = len(samples)
    modality_samples_list = []
    
    for label in range(num_labels):
        modality_samples_list.append(F.normalize(samples[label][modality], dim=-1))
    
    modality_samples = torch.concat(modality_samples_list)
    samples_cnt = [ms.shape[0] for ms in modality_samples_list]
    
    np.savez("modality_samples.npz", **{str(lbl): ms.cpu().numpy() for lbl, ms in enumerate(modality_samples_list)})

    reduced_emb = reduction_model.fit_transform(modality_samples.cpu().numpy())
    cnt = 0
    for label in range(num_labels):
        sc = samples_cnt[label]
        x = reduced_emb[cnt: cnt+sc+1, 0]
        y = reduced_emb[cnt: cnt+sc+1, 1]
        plt.scatter(x, y, label=f"label: {label}")
        cnt += sc
    plt.legend()
    plt.show()
    return reduced_emb


if __name__ == '__main__':

    # load opts
    with open('config/eval_config.yaml') as f:
        opts = AttrDict(yaml.safe_load(f))

    # set device
    device = opts.eval.device
    if not torch.cuda.is_available():
        assert 'cuda' not in opts.eval.device, "CUDA is not available."

    # load model
    model = MMEncoder(opts)
    model.eval()
    model.to(device)

    # load ckpt
    if opts.eval.load_ckpt:
        ckpt = torch.load(opts.eval.load_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print("Checkpoint load complete.")


    # dataloader for each class
    loaders = []
    for data_path in opts.eval.subset_config_path:
        with open(data_path) as f:
            data_opts = AttrDict(yaml.safe_load(f))
        dataset = MMDataset(data_opts)
        loader = DataLoader(
            dataset,
            batch_size=opts.eval.batch_size,
        )
        loaders.append(loader)

    samples = {
        lbl: {
            'video': torch.empty(0, opts.model.emb_dim).to(device), 
            'spec': torch.empty(0, opts.model.emb_dim).to(device),
            'audio': torch.empty(0, opts.model.emb_dim).to(device)
        } 
        for lbl in range(len(loaders))
    }


    prog_bar = tqdm(range(sum([len(loader) for loader in loaders])))

    for lbl_idx, loader in enumerate(loaders):

        # inference and get embeddings
        for batch in loader:
            # load batch to device
            vid_inp = batch['video'].to(device)
            spec_inp = batch['spec'].to(device)
            aud_inp = batch['audio'].to(device)

            with torch.no_grad():
                emb = model(vid_inp, spec_inp, aud_inp)
                samples[lbl_idx]['video'] = torch.concat([samples[lbl_idx]['video'], emb['video']], dim=0)
                samples[lbl_idx]['spec'] = torch.concat([samples[lbl_idx]['spec'], emb['spec']], dim=0)
                samples[lbl_idx]['audio'] = torch.concat([samples[lbl_idx]['audio'], emb['audio']], dim=0)
            prog_bar.update()
    reduction_model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50.)
    # # reduction_model = PCA(n_components=2)
    reduced_emb = analyze_embedding('spec', samples, reduction_model)


