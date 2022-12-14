import os
import json
import torch
import torchvision
import torchaudio
import librosa
import soundfile

from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm import tqdm


def set_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default="./raw_data/data/balanced_train_segments"
    )
    parser.add_argument(
        '--proc_data_dir',
        type=str,
        default="./processed_data"
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=15
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000
    )
    parser.add_argument(
        '--video_ext',
        type=str,
        default=".mp4"
    )
    parser.add_argument(
        '--audio_ext',
        type=str,
        default=".flac"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )
    args = parser.parse_args()
    return args


def extract_video(
    vid_path,
    start_sec,
    end_sec,
    num_frame=15, 
    data_dir=None
):
    
    # vid: (T, H, W, C)
    vid, _, _ = torchvision.io.read_video(vid_path, start_sec, end_sec, pts_unit='sec')
    assert vid.shape[0] >= num_frame, f"Video should be longer than num_frame {num_frame}"
    
    stride = vid.shape[0] // num_frame
    vid = vid[::stride][:num_frame]
    assert vid.shape[0] == num_frame
    
    if data_dir is not None:
        # save file as video_name_sid
        # save info of the file in meta.json
        meta_info = {
            'start': f'{start_sec:.3f}',
            'end': f'{end_sec:.3f}',
            'fps': num_frame / (end_sec - start_sec)
        }
        vid_name = os.path.basename(vid_path)
        save_processed_video(vid, vid_name, data_dir, meta_info)
        
    return vid

        
def save_processed_video(
    vid,
    vid_name,
    data_dir,
    meta_info
):
    
    # # open meta.json
    # meta_path = os.path.join(data_dir, 'meta.json')
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'r') as f:
    #         meta = json.load(f)
    # else:
    #     meta = {}
    
    # if vid_name in meta:
    #     vid_sid = len(meta[vid_name])    # get segment id
    # else:
    #     meta[vid_name] = {}
    #     vid_sid = 0
    
    # # write video file
    # assert len(str(vid_sid)) <= 2, f"vid_sid number cannot be bigger than 99"
    
    # name, ext = os.path.splitext(vid_name)
    # vid_path = os.path.join(data_dir, name + f'_{vid_sid:02}' + ext)
    vid_path = os.path.join(data_dir, vid_name)

    # write video segment
    torchvision.io.write_video(vid_path, vid, meta_info['fps'])
    
    # # write meta.json
    # meta[vid_name][vid_sid] = meta_info
    # with open(meta_path, 'w') as f:
    #     json.dump(meta, f, indent=4)
    
    
def extract_audio(
    aud_path,
    start_sec,
    end_sec,
    sr=16000,
    data_dir=None
):
    
    # wav: (num_sample, )
    aud, _ = librosa.load(aud_path, sr=sr, offset=start_sec, duration=(end_sec - start_sec))
    assert len(aud) == (end_sec - start_sec) * sr, f"Audio is not {end_sec - start_sec}sec long!, {aud_path}"
    
    if data_dir is not None:
        # save file as audio_name_sid
        # save info of the file in meta.json
        meta_info = {
            'start': f'{start_sec:.3f}',
            'end': f'{end_sec:.3f}',
            'sr': sr
        }
        aud_name = os.path.basename(aud_path)
        save_processed_audio(aud, aud_name, data_dir, meta_info)
        
    return aud

        
def save_processed_audio(
    aud,
    aud_name,
    data_dir,
    meta_info
):
    
    # # open meta.json
    # meta_path = os.path.join(data_dir, 'meta.json')
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'r') as f:
    #         meta = json.load(f)
    # else:
    #     meta = {}
    
    # if aud_name in meta:
    #     aud_sid = len(meta[aud_name])    # get segment id
    # else:
    #     meta[aud_name] = {}
    #     aud_sid = 0
    
    # # write video file
    # assert len(str(aud_sid)) <= 2, f"vid_sid number cannot be bigger than 99"
    
    # name, ext = os.path.splitext(aud_name)
    # aud_path = os.path.join(data_dir, name + f'_{aud_sid:02}' + ext)
    aud_path = os.path.join(data_dir, aud_name)

    # write audio segment
    soundfile.write(aud_path, aud, meta_info['sr'])
    
    # # write meta.json
    # meta[aud_name][aud_sid] = meta_info
    # with open(meta_path, 'w') as f:
    #     json.dump(meta, f, indent=4)
    
    
    
    
if __name__ == '__main__':
    args = set_args()

    # create folders
    os.makedirs(os.path.join(args.proc_data_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(args.proc_data_dir, 'spec'), exist_ok=True)
    os.makedirs(os.path.join(args.proc_data_dir, 'video'), exist_ok=True)

    # directory path to raw data
    aud_dir = os.path.join(args.raw_data_dir, 'audio')
    vid_dir = os.path.join(args.raw_data_dir, 'video')

    # only take paired data
    vid_fn_list = [os.path.splitext(vid_f)[0] for vid_f in os.listdir(vid_dir) if args.video_ext in vid_f]
    aud_fn_list = [os.path.splitext(aud_f)[0] for aud_f in os.listdir(aud_dir) if args.audio_ext in aud_f]
    paired = list(set().intersection(vid_fn_list, aud_fn_list))
    

    # preprocess function to pass on multiprocess
    def process_pair(file_name):

        vp = os.path.join(vid_dir, file_name + args.video_ext)
        ap = os.path.join(aud_dir, file_name + args.audio_ext)

        # video
        extract_video(
            vid_path=os.path.join(vid_dir, file_name + args.video_ext),
            start_sec=5,
            end_sec=8,
            num_frame=15,
            data_dir=os.path.join(args.proc_data_dir, 'video')
        )
        # spectrogram
        extract_audio(
            aud_path=os.path.join(aud_dir, file_name + args.audio_ext), 
            start_sec=5,
            end_sec=8,
            sr=args.sample_rate,
            data_dir=os.path.join(args.proc_data_dir, 'spec')
        )
        # audio
        extract_audio(
            aud_path=os.path.join(aud_dir, file_name + args.audio_ext), 
            start_sec=2,
            end_sec=5,
            sr=args.sample_rate,
            data_dir=os.path.join(args.proc_data_dir, 'audio')
        )


    # multiprocess
    with Pool(args.num_workers) as p:
        p.map(process_pair, tqdm(paired), total=len(paired))

    # for fn in tqdm(paired):
    #     vp = os.path.join(vid_dir, fn + args.video_ext)
    #     ap = os.path.join(aud_dir, fn + args.audio_ext)
    #     extract_audio(ap, 0, 3, args.sample_rate, os.path.join(args.proc_data_dir, 'audio'))
    #     extract_audio(ap, 3, 6, args.sample_rate, os.path.join(args.proc_data_dir, 'spec'))
    #     extract_video(vp, 3, 6, args.num_frames, os.path.join(args.proc_data_dir, 'video'))


    # aud_dir = os.path.join(args.raw_data_dir, 'audio')
    # aud_fps = [
    #     os.path.join(aud_dir, aud_f) for aud_f in os.listdir(aud_dir)
    #     if os.path.splitext(aud_f)[1] in ['.wav', '.flac']
    # ]

    # vid_dir = os.path.join(args.raw_data_dir, 'video')
    # vid_fps = [
    #     os.path.join(vid_dir, vid_f) for vid_f in os.listdir(vid_dir)
    #     if os.path.splitext(vid_f)[1] in ['.mp4', '.avi']
    # ]
    
    # for vid_fp, aud_fp in tqdm(list(zip(vid_fps, aud_fps))):
    #     extract_audio(aud_fp, 0, 3, data_dir=os.path.join(args.proc_data_dir, 'audio'))    # audio
    #     extract_audio(aud_fp, 3, 6, data_dir=os.path.join(args.proc_data_dir, 'spec'))    # spectrogram
    #     extract_video(vid_fp, 3, 6, data_dir=os.path.join(args.proc_data_dir, 'video'))    # video
        
