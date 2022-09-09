# load resnet50

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

from external.temporal_shift import make_temporal_shift
from external.models import Cnn14, Res1dNet31


class VideoEncoder(nn.Module):
    def __init__(
        self,
        num_segments=8,    # number of frames
        n_div=8,           # default value of TSM
        # dropout=0.5,
    ):
        super(VideoEncoder, self).__init__()
        self.num_segments = num_segments
        self.n_div = n_div
        
        # create base model
        self.base_model = resnet50(pretrained=False)
        # setattr(self.base_model, 'fc', nn.Dropout(p=dropout))
        setattr(self.base_model, 'fc', nn.Identity())
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
    def __init__(
        self,
        sample_rate=32000,
        window_size_ms=0.025,
        hop_size_ms=0.010,
        mel_bins=64,
        fmin=50,
        fmax=14000
    ):
        super(SpectrogramEncoder, self).__init__()
        self.sample_rate = sample_rate
        
        # create base model
        self.base_model = Cnn14(
            sample_rate=sample_rate,
            window_size=int(sample_rate * window_size_ms),    # 25 ms
            hop_size=int(sample_rate * hop_size_ms),       # 10 ms
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=1
        )
        
    def forward(self, inp):
        # inp: (n_batch, audio_len)
        
        # out: (n_batch, emb_dim)
        out = self.base_model(inp)['embedding']
        
        return out
    
    
class AudioEncoder(nn.Module):
    def __init__(
        self,
        sample_rate=32000
    ):
        super(AudioEncoder, self).__init__()
        self.sample_rate = sample_rate
        
        # create base model
        # parameter does not matter
        self.base_model = Res1dNet31(
            sample_rate=sample_rate,
            window_size=int(sample_rate * 0.025),    # 25 ms
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


class MMEncoder(nn.Module):
    def __init__(self, opts):
        super(MMEncoder, self).__init__()
        self.vid_enc = VideoEncoder(
            opts.model.video.num_segments,
            opts.model.video.n_div,
        )
        self.spec_enc = SpectrogramEncoder(
            opts.model.spec.sample_rate,
            opts.model.spec.window_size_ms,
            opts.model.spec.hop_size_ms,
            opts.model.spec.mel_bins,
            opts.model.spec.fmin,
            opts.model.spec.fmax,
        )
        self.aud_enc = AudioEncoder(
            opts.model.audio.sample_rate,
        )
        
        self.projector = nn.Sequential(
            nn.Linear(opts.model.feature_dim, opts.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(opts.model.hidden_dim, opts.model.emb_dim)
        )
        self.projector.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.)
        
    def forward(
        self,
        vid_inp,
        spec_inp,
        aud_inp
    ):
        vid_out = self.projector(self.vid_enc(vid_inp))
        spec_out = self.projector(self.spec_enc(spec_inp))
        aud_out = self.projector(self.aud_enc(aud_inp))
        
        emb_dict = {
            'video': vid_out,
            'spec': spec_out,
            'audio': aud_out
        }
        return emb_dict
        


if __name__ == '__main':
    
    video_encoder = VideoEncoder(8)
    # TODO: test model parameters

