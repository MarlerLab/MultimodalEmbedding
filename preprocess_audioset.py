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


MIN_DATA_LEN = 9.0    # sec


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
    parser.add_argument(
        '--validate_files',
        action='store_true'
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
    vid_path = os.path.join(data_dir, vid_name)

    # write video segment
    torchvision.io.write_video(vid_path, vid, meta_info['fps'])
    
    
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
    aud_path = os.path.join(data_dir, aud_name)

    # write audio segment
    soundfile.write(aud_path, aud, meta_info['sr'])
    

# https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def video_stat(filename):
    import subprocess, json

    result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True).decode()
    if not json.loads(result)['streams']:
        return 0, 0
    fields = json.loads(result)['streams'][0]

    duration = fields['duration']
    fps      = eval(fields['r_frame_rate'])
    return duration, fps
    
    
    
if __name__ == '__main__':
    args = set_args()

    # create folders
    os.makedirs(os.path.join(args.proc_data_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(args.proc_data_dir, 'spec'), exist_ok=True)
    os.makedirs(os.path.join(args.proc_data_dir, 'video'), exist_ok=True)

    # directory path to raw data
    aud_dir = os.path.join(args.raw_data_dir, 'audio')
    vid_dir = os.path.join(args.raw_data_dir, 'video')

    if args.validate_files:
        # delete corrupted files
        os.makedirs(os.path.join(args.raw_data_dir, 'corrupted_audio'), exist_ok=True)
        os.makedirs(os.path.join(args.raw_data_dir, 'corrupted_video'), exist_ok=True)

        print("Deleting corrupte audio files...")
        for aud_f in tqdm(os.listdir(aud_dir)):
            if args.audio_ext in aud_f:
                aud_fp = os.path.join(aud_dir, aud_f)
                duration = librosa.get_duration(filename=aud_fp)
                if duration < MIN_DATA_LEN:
                    print(aud_fp)
                    os.rename(aud_fp, os.path.join(
                        args.raw_data_dir,
                        'corrupted_audio',
                        aud_f
                    ))

        print("Deleting corrupted video files...")
        for vid_f in tqdm(os.listdir(vid_dir)):
            if args.video_ext in vid_f:
                vid_fp = os.path.join(vid_dir, vid_f)
                duration, _ = video_stat(vid_fp)
                if float(duration) < MIN_DATA_LEN:
                    print(vid_fp)
                    os.rename(vid_fp, os.path.join(
                        args.raw_data_dir,
                        'corrupted_video',
                        vid_f
                    ))


    # only take paired data
    vid_fn_list = {os.path.splitext(vid_f)[0] for vid_f in os.listdir(vid_dir) if args.video_ext in vid_f}
    aud_fn_list = {os.path.splitext(aud_f)[0] for aud_f in os.listdir(aud_dir) if args.audio_ext in aud_f}
    paired = list(set.intersection(vid_fn_list, aud_fn_list))
    

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


    # # multiprocess
    # with Pool(args.num_workers) as p:
    #     p.map(process_pair, tqdm(paired, total=len(paired)))

    for fn in tqdm(paired):
        process_pair(fn)

