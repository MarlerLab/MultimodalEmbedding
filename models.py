# load resnet50

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

from external.temporal_shift import make_temporal_shift


class VideoEncoder(nn.Module):
    def __init__(
        self,
        num_segments=8,    # number of frames
        n_div=8,           # default value of TSM
        dropout=0.5,
    ):
        super(VideoEncoder, self).__init__()
        self.num_segments = num_segments
        self.n_div = n_div
        
        # create base model
        self.base_model = resnet50(pretrained=False)
        setattr(self.base_model, 'fc', nn.Dropout(p=0.5))
        make_temporal_shift(self.base_model, num_segments, n_div)
    
    
    def forward(self, inp):
        # inp: (n_batch, num_segments, c, h, w)
        num_segments = inp.shape[1]
        assert self.num_segments == num_segments, "Invalid number of frames"
        
        # inp: (n_batch * num_segments, c, h, w)
        inp = inp.view(-1, *inp.shape[-3: ])
        
        # out: (n_batch * num_segments, emb_dim)
        out = self.base_model(inp)
        
        # out: (n_batch, num_segments, emb_dim)
        out = out.view(-1, num_segments, out.shape[1])
        
        # pooling
        # out: (n_batch, emb_dim)
        out = out.mean(1, keepdim=False)
        
        return out
    

class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.base_model = Cnn14(
            sample_rate=32000,
            window_size=int(sample_rate * 0.020),    # 20 ms
            hop_size=int(sample_rate * 0.010),       # 10 ms
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=1
        )
        
    def forward(self, inp):
        # inp: (n_batch, audio_len)
        
        # out: (n_batch, emb_dim)
        out = self.base_model(inp)['embedding']
        
        return out
    
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.base_model = Res1dNet31(
            sample_rate=32000,
            window_size=int(sample_rate * 0.020),    # 20 ms
            hop_size=int(sample_rate * 0.010),       # 10 ms
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=1
        )
        
    def forward(self, inp):
        # inp: (n_batch, audio_len)
        
        # out: (n_batch, emb_dim)
        out = self.base_model(inp)['embedding']
        
        return out
        
        

if __name__ == '__main':
    
    video_encoder = VideoEncoder(8)

