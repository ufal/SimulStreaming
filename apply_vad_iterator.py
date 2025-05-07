# for debugging only

from whisper_streaming.whisper_online_main import load_audio_chunk, load_audio

import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--vac-chunk-size', type=float, default=0.04, 
                    help='VAC sample size in seconds.')
parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')


args = parser.parse_args()


SAMPLING_RATE = 16000
duration = len(load_audio(args.audio_path))/SAMPLING_RATE

from whisper_streaming.silero_vad_iterator import FixedVADIterator
import torch
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
vac = FixedVADIterator(model)



b, e = args.start_at, args.start_at+args.vac_chunk_size

ret_beg = None
while e <= duration:
    audio = load_audio_chunk(args.audio_path, b, e)

    x = vac(audio,return_seconds=True)
    if x is not None:
        print(b,e,x,file=sys.stderr)
        if 'start' in x:
            ret_beg = x['start']
        if 'end' in x:  
            print(ret_beg, x['end'], "segment",sep="\t")
    b = e
    e += args.vac_chunk_size