import os
from typing import List, Tuple
from pathlib import Path
import torchaudio
import torch
import torch.nn as nn
from torch.utils import data
import json
import torchaudio.functional as F
import torchaudio.transforms as T
from src.utils import project_root
from src.data.utils import ru_alphabet


class ruLibriSpeechDataset(data.Dataset):

    def __init__(self, data_path: Path, data_type: str = "train", sample_rate: int = 16000):
        assert data_type in ["train", "test"]
        self.data_path = Path(data_path) / data_type
        self.sample_rate = sample_rate
        
        self.manifest = []
        path = self.data_path / "manifest.json" 
        with open(path, 'r') as f:
            for line in f.readlines():
                self.manifest.append(json.loads(line))

    def __getitem__(self, index: int):
        item = self.manifest[index]
        wav, sr = torchaudio.load(self.data_path / item["audio_filepath"])
        audio = F.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        text = item["text"] 

        return {
            "audio": audio,
            "text": text
        }

    def __len__(self):
        return len(self.manifest)


class TextEncoder:

    def __init__(self, alphabet, blank=0):
        self.alphabet = alphabet
        self.blank = blank
        
    def encode(self, text: str) -> List[int]:
        return list(map(lambda x: ru_alphabet[x], text))
    
    def decode(self, text: List[int]) -> str:
        return "".join(list(map(lambda x: ru_alphabet.inverse[int(x)], text))).replace(ru_alphabet.inverse[self.blank], "")
    

class Collator:
    
    def __init__(self, encoder, max_length=45000):
        self.encoder = encoder
        self.max_length = max_length
   #     self.min_length = 10000
        
    def __call__(self, batch):
        # we pad wavs to one length
        lengths = []
        
        for item in batch:
            lengths.append(item["audio"].size(-1))
        
        batch_wavs = torch.zeros(len(batch), min(max(lengths), self.max_length))
        for i, item in enumerate(batch):
            batch_wavs[i, :min(lengths[i], self.max_length)] = item["audio"][..., :self.max_length]
        
        lengths = torch.tensor(lengths).long()
        
        # we encode texts and pad them too
        targets = []
        target_lengths = []
        
        for i, item in enumerate(batch):
            text = self.encoder.encode(item["text"])
            if lengths[i] > self.max_length:
                text = text[:int(len(text) * self.max_length / lengths[i])]
            targets.append(text)
            target_lengths.append(len(text))
        
        lengths = torch.clamp(lengths, max=self.max_length)

        batch_targets = torch.zeros(len(batch), max(target_lengths)).long()
        
        for i, target in enumerate(targets):
            assert target_lengths[i] > 1, "aaaaaa"
            batch_targets[i, :target_lengths[i]] = torch.tensor(target).long()
            
        target_lengths = torch.LongTensor(target_lengths)
        
        
        return {
            'wav': batch_wavs,
            'length': lengths,
            'targets': batch_targets,
            'target_lengths': target_lengths
        }


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


class Featurizer(nn.Module):
    
    def __init__(self, n_mels=64, sample_rate=16000):
        super(Featurizer, self).__init__()
        self.sample_rate = sample_rate
        
        self.featurizer = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=128,
            n_mels=n_mels,
            center=True,
            onesided=True,
            normalized=False
        )
        # self.normalizer = NormalizedSpectrogram(
        #     normalize="touniform"
        # )
        
    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        mel_spectrogram = mel_spectrogram.clamp(min=1e-8).log()
        # mel_spectrogram = self.normalizer(mel_spectrogram)
        
        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            # length += 1 + 4
            
            return mel_spectrogram, length
        
        return mel_spectrogram


class FeaturizerMFCC(nn.Module):
    
    def __init__(self, n_mfcc=64, sample_rate=16000):
        super(Featurizer, self).__init__()
        self.sample_rate = sample_rate
        
        self.featurizer = T.MFCC(
            sample_rate=self.sample_rate, 
            n_mfcc=n_mfcc, 
            log_mels = True)
        
    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        
        if length is not None:
            length = (length - self.featurizer.MelSpectrogram.win_length) // self.featurizer.MelSpectrogram.hop_length
            return mel_spectrogram, length
        
        return mel_spectrogram
