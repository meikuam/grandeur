from typing import List, Tuple
from pathlib import Path
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from src.utils import project_root
import numpy as np

# free spoken digits
class FreeSpokenDigits(Dataset):
    _sample_rate = 8000
    def __init__(self, path: Path):
        self.data_path = Path(path) / "recordings"
        self.paths = list(self.data_path.glob('**/*.wav'))

    def __getitem__(self, index: int):
        path = self.paths[index]
        wav, sr = torchaudio.load(path)
        label = int(path.stem.split("_")[0])
        return wav, label

    def __len__(self):
        return len(self.paths)


class Collator:
    def __call__(self, batch: List[Tuple[torch.Tensor, int]]):
        lengths = []
        wavs, labels = zip(*batch)
        
        for wav in wavs:
            lengths.append(wav.size(-1))
        
        # here we should pad wvas to one length, cause we need pass it to network
        batch_wavs = torch.zeros(len(batch), max(lengths))
        for i, wav in enumerate(wavs):
            batch_wavs[i, :lengths[i]] = wav
        
        
        labels = torch.tensor(labels).long()
        lengths = torch.tensor(lengths).long()

        labels.pin_memory()
        
        return {
            'wav': batch_wavs,
            'label': labels,
            'length': lengths,
        }


class SqueezeformerCollator:
    def __call__(self, batch: List[Tuple[torch.Tensor, int]]):
        lengths = []
        wavs, labels = zip(*batch)
        
        for wav in wavs:
            lengths.append(wav.size(-1))
        
        # here we should pad wvas to one length, cause we need pass it to network
        batch_wavs = torch.zeros(len(batch), max(max(lengths),64))
        for i, wav in enumerate(wavs):
            batch_wavs[i, :lengths[i]] = wav
        
        
        lengths = torch.tensor(lengths).long()
        
        targets = torch.tensor(labels).long()
        target_lengths = torch.LongTensor([1] * len(wavs))
        
        
        return {
            'wav': batch_wavs,
            'length': lengths,
            'targets': targets,
            'target_lengths': target_lengths
        }


class Featurizer(nn.Module):
    
    def __init__(self):
        super(Featurizer, self).__init__()
        
        self.featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=8_000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            center=True
        )
        
    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        mel_spectrogram = mel_spectrogram.clamp(min=1e-5).log()
        
        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            length += 1 + 4
            
            return mel_spectrogram, length
        
        return mel_spectrogram


from torchvision.transforms import Normalize
class NormalizedSpectrogram(torch.nn.Module):
    def __init__(self, normalize=None, *args, **kwargs):
        super(NormalizedSpectrogram, self).__init__(*args, **kwargs)
        if normalize == 'to05':
            self.normalize = Normalize([0.5], [0.5])
        elif normalize == 'touniform':
            self.normalize = lambda x: (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-18)
        else:
            self.normalize = None


    def forward(self, melsec):
        if self.normalize is not None:
            logmelsec = torch.log(torch.clamp(melsec, min=1e-18))
            melsec = self.normalize(logmelsec)
        return melsec

class SqueezeformerFeaturizer(nn.Module):
    
    def __init__(self):
        super(SqueezeformerFeaturizer, self).__init__()
        
        self.featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=8_000,
            n_fft=512,
            win_length=128,
            hop_length=64,
            n_mels=64,
            center=True,
            # pad=2
        )
        self.normalizer = NormalizedSpectrogram(
            normalize="touniform"
        )
        
    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        # mel_spectrogram = mel_spectrogram.clamp(min=1e-5).log()
        mel_spectrogram = self.normalizer(mel_spectrogram)
        
        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            # length += 1 + 4 #+ 2
            
            return mel_spectrogram, length
        
        return mel_spectrogram

