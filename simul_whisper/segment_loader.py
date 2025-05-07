import jsonlines

import torch
import torchaudio

from ..whisper.audio import N_FFT, HOP_LENGTH, SAMPLE_RATE
from .config import AlignAttConfig


class Segment:
    def __init__(self, audio_path, samples_to_read, samples_in_chunk):
        self.audio_path = audio_path
        audio, sr = torchaudio.load(audio_path, normalize=True)
        self.audio = audio.squeeze()
        self.audio_len_s = self.audio.shape[0] / SAMPLE_RATE
        assert sr == SAMPLE_RATE
        self.samples_to_read = samples_to_read
        self.samples_in_chunk = samples_in_chunk
        self.buffer_len = samples_in_chunk - samples_to_read

    def __iter__(self):
        frames_in_chunk = self.audio[:self.samples_in_chunk]
        read_pointer = frames_in_chunk.shape[0]
        yield frames_in_chunk, (read_pointer >= self.audio.shape[0])
        while read_pointer < self.audio.shape[0]:
            frames_in_chunk = torch.cat(
                (frames_in_chunk[-self.buffer_len:], self.audio[read_pointer:read_pointer+self.samples_to_read]),
                dim=0
                )
            read_pointer += self.samples_to_read
            yield frames_in_chunk, (read_pointer >= self.audio.shape[0])


class SegmentWrapper(Segment):
    def __init__(self, audio_path, segment_length):
        frames_to_read = int((segment_length * SAMPLE_RATE) / HOP_LENGTH)
        samples_to_read = frames_to_read * HOP_LENGTH
        samples_in_chunk = samples_to_read + N_FFT - HOP_LENGTH
        super().__init__(
            audio_path, 
            samples_to_read=samples_to_read, 
            samples_in_chunk=samples_in_chunk)


class SegmentLoader:

    def __init__(self, cfg: AlignAttConfig):
        self.cfg = cfg
        frames_to_read = int((cfg.segment_length * SAMPLE_RATE) / HOP_LENGTH)
        self.samples_to_read = frames_to_read * HOP_LENGTH
        self.samples_in_chunk = self.samples_to_read + N_FFT - HOP_LENGTH
        with open(cfg.eval_data_path) as f:
            self.data_list = [l for l in jsonlines.Reader(f)]
    
    def __getitem__(self, i):
        return Segment(
            audio_path=self.data_list[i]["audio"], 
            samples_to_read=self.samples_to_read,
            samples_in_chunk=self.samples_in_chunk
        ), self.data_list[i]["sentence"]
    
    def __len__(self) -> int:
        return len(self.data_list)
    