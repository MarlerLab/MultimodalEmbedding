import torch
import torch.nn as nn

from info_nce import InfoNCE


class MMConLoss(nn.Module):
    """Multi-modal contrastive loss. ex) video, audio, spectrogram"""
    def __init__(
        self,
        modalities=['video', 'audio', 'spec'],
        temperature=0.1
    ):
        super(MMConLoss, self).__init__()
        self.info_nce = InfoNCE(temperature=temperature, reduction='mean')
        num_modalities = len(modalities)
        self.modality_pairs = []
        self.modality_pairs += [
            (modalities[i], modalities[(i+1)%num_modalities])
            for i in range(num_modalities)
        ]
        self.modality_pairs += [
            (modalities[i], modalities[(i-1)%num_modalities]) 
            for i in range(num_modalities)
        ]
        
        
    def forward(self, emb):
        # emb[modality]: (n_batch, emb_dim)
        assert type(emb) == dict

        loss = 0
        
        for query, key in self.modality_pairs:
            loss += self.info_nce(emb[query], emb[key])
        
        return loss
    