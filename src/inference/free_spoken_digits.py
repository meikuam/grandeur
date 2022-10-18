import sys
sys.path.append("..")

import torch.multiprocessing as mp
mp.set_start_method('spawn')
import torch
from torch.utils.data import DataLoader, Subset
from src.utils import project_root
from src.data.free_spoken_digits import Collator, Featurizer

import torchaudio
import torchaudio.functional as F

if __name__ == "__main__":
    model_path = project_root()/"ckpt/free_spoken_digits.ckpt"
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    loaded_model = torch.load(model_path, map_location=torch.device("cpu")).to(device)
    featurizer = Featurizer().to(device)

    
    # infer_audio_path = project_root()/"data/five.m4a"
    infer_audio_path = project_root()/"data/one.m4a"
    

    wav, sr = torchaudio.load(infer_audio_path)

    resampled_wav = F.resample(wav, sr, featurizer.featurizer.sample_rate)

    resampled_wav = resampled_wav.to(device)
    length = resampled_wav.size(-1)

    mel_inputs, length = featurizer(resampled_wav, length)
    length = torch.tensor([length]).long().to(device)

    outputs = loaded_model(mel_inputs, length)
    classes = outputs.argmax(-1)
    for i, prob in enumerate(outputs[0].tolist()):
        print(i, prob)
