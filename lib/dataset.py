import bisect
import csv
import functools
import os
import random

import numpy as np

import torch
import torchaudio
torchaudio.set_audio_backend('soundfile')

from torch.nn import functional as F
from torch.utils.data import Dataset
from torchaudio import sox_effects, transforms


class DefaultSet(Dataset):
    def __init__(self, root, subset, input_len, n_fft, sample_rate=None):
        super().__init__()

        self.sample_rate = 44100 if sample_rate is None else sample_rate
        self.input_len = input_len
        self.fft = transforms.Spectrogram(n_fft=n_fft)

        data = pd.read_csv(os.path.join(root, subset + '.csv'))
        self.files = tuple(data['vocal'])
        labels = tuple(data['label'])

        uniq_labels = sorted(set(labels))
        self.num_classes = len(uniq_labels)

        self.labels = [bisect.bisect_left(uniq_labels, label) for label in labels]

    def load(self, index):
        audio, sample_rate = torchaudio.load(self.files[index])
        if sample_rate != 44100:
            transform = transforms.Resample(sample_rate, 44100)
            audio = transform(audio)
            sample_rate = 44100
        assert sample_rate == self.sample_rate
        return torch.unsqueeze(torch.mean(audio, axis=0), dim=0)  # make it mono

    def reshape(self, audio, length):
        current = audio.shape[1]
        if current < length:
            audio = F.pad(audio, (0, length - current))
        elif current > length:
            idx = random.randint(0, current - length)
            audio = audio[:, idx: idx + length]
        return audio

    def __getitem__(self, index):
        audio = self.reshape(self.load(index), self.input_len)
        return index, self.fft(audio), torch.tensor(self.labels[index], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.files)


class ContrastiveSet(DefaultSet):
    def __init__(self, root, subset, input_len, n_fft, pitch, stretch, sample_rate=None):
        super().__init__(root, subset, input_len, n_fft, sample_rate)
        self.pitch, self.stretch = pitch, stretch

    def pitch_shift(self, audio, pitch):
        source = self.reshape(audio, self.input_len + 50)

        effects = [['pitch', str(pitch * 100)], ['rate', str(self.sample_rate)]]
        target, sample_rate = sox_effects.apply_effects_tensor(source, self.sample_rate, effects)
        assert sample_rate == self.sample_rate

        return self.reshape(target, self.input_len)

    def time_stretch(self, audio, speed):
        source = self.reshape(audio, int(self.input_len * speed) + 50)

        effects = [['tempo', str(speed)], ['rate', str(self.sample_rate)]]
        target, sample_rate = sox_effects.apply_effects_tensor(source, self.sample_rate, effects)
        assert sample_rate == self.sample_rate

        return self.reshape(target, self.input_len)

    def __getitem__(self, index):
        audio = self.load(index)
        audio_orig = self.reshape(audio, self.input_len)

        reshape_func = functools.partial(self.reshape, length=self.input_len)
        pitch_func = functools.partial(self.pitch_shift, pitch=random.choice([-3, 3]))
        stretch_func = functools.partial(self.time_stretch, speed=random.choice([0.65, 1.70]))

        pos_funcs = []
        if self.pitch == 'pos':
            pos_funcs.append(pitch_func)
        if self.stretch == 'pos':
            pos_funcs.append(stretch_func)
        if len(pos_funcs) == 0:
            pos_funcs.append(reshape_func)
        audio_pos = random.choice(pos_funcs)(audio)

        neg_funcs = []
        if self.pitch == 'neg':
            neg_funcs.append(pitch_func)
        if self.stretch == 'neg':
            neg_funcs.append(stretch_func)
        if len(neg_funcs) == 0:
            neg_funcs.append(lambda x: audio_orig)  # not used
        audio_neg = random.choice(neg_funcs)(audio)

        audio_orig = self.fft(audio_orig)
        audio_pos = self.fft(audio_pos)
        audio_neg = self.fft(audio_neg)

        return index, audio_orig, audio_pos, audio_neg
