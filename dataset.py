# dataloader
import os
import torch
import torchvision
import torchvision.transforms as vision_T
import torchaudio

from torch.utils.data import Dataset
from tqdm import tqdm

# TODO: refactor this disgusting code

class MMDataset(Dataset):
    def __init__(self, opts):
        self.video_dir = opts.dataset.video_dir
        self.spec_dir = opts.dataset.spec_dir
        self.audio_dir = opts.dataset.audio_dir
        
        self.v_ext = opts.dataset.video_ext
        self.s_ext = opts.dataset.spec_ext
        self.a_ext = opts.dataset.audio_ext
        
        v_fns = [os.path.splitext(fn)[0] for fn in os.listdir(self.video_dir) if os.path.splitext(fn)[1] == self.v_ext]
        s_fns = [os.path.splitext(fn)[0] for fn in os.listdir(self.spec_dir) if os.path.splitext(fn)[1] == self.s_ext]
        a_fns = [os.path.splitext(fn)[0] for fn in os.listdir(self.audio_dir) if os.path.splitext(fn)[1] == self.a_ext]

        self.file_names = list(set.intersection(set(v_fns), set(s_fns), set(a_fns)))
        
        self.v_transform = vision_T.Compose([
            vision_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            vision_T.RandomResizedCrop((50, 50)),
            vision_T.ColorJitter(0.5, 0.5, 0.5, 0.5)
        ])
        
        # Pre-load everything to RAM to avoid dataloader bottleneck
        self.preload = opts.dataset.preload
        if self.preload:
            raise NotImplementedError("Implmentation needs to change to cache the entire datset!")
            print("Caching data to RAM...")
            self.cache = {
                # 'video_path': video,
                # 'spec_path': spec,
                # 'audio_path': audio
            }
            for file_name in tqdm(self.file_names):
                vp = os.path.join(self.video_dir, file_name + self.v_ext)
                sp = os.path.join(self.spec_dir, file_name + self.s_ext)
                ap = os.path.join(self.audio_dir, file_name + self.a_ext)

                self.cache[vp] = torchvision.io.read_video(vp, pts_unit='sec')[0].numpy()
                self.cache[sp] = torchaudio.load(sp)[0].numpy()
                self.cache[ap] = torchaudio.load(ap)[0].numpy()

        
    def _load_video(self, video_path):
        
        # video: (T, H, W, C)
        if self.preload:
            video = torch.from_numpy(self.cache[video_path])
        else:
            video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            
        if video.dtype == torch.uint8:
            video = video / 255
        
        # video: (T, C, H, W)    to pass transform
        video = video.permute(0, 3, 1, 2)
        # video: (T, C, H', W')
        video = self.v_transform(video)
        return video
    
    def _load_audio(self, audio_path, reduction='mean'):
        assert reduction in ['mean', 'first']
        
        # audio: (C, num_samples)
        if self.preload:
            audio = torch.from_numpy(self.cache[audio_path])
        else:
            audio, sr = torchaudio.load(audio_path)
        
        # audio: (num_samples, )
        if reduction == 'mean':
            audio = audio.mean(dim=0, keepdim=False)
        elif reduction == 'first':
            audio = audio[0]
        
        return audio
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        vp = os.path.join(self.video_dir, file_name + self.v_ext)
        sp = os.path.join(self.spec_dir, file_name + self.s_ext)
        ap = os.path.join(self.audio_dir, file_name + self.a_ext)
        
        v_data = self._load_video(vp)
        s_data = self._load_audio(sp)    # audio as input for spectrogram
        a_data = self._load_audio(ap)

        
        data = {
            'video': v_data,   # (T, C, H', W')
            'spec': s_data,    # (num_samples, )
            'audio': a_data    # (num_samples, )
        }
        # data = {
        #     'video': self.cache['video'][idx],   # (T, C, H', W')
        #     'spec': self.cache['spec'][idx],    # (num_samples, )
        #     'audio': self.cache['audio'][idx]    # (num_samples, )
        # }
        
        return data
