import os
import sys
sys.path.append(".")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pathlib
from tqdm import tqdm
from itertools import islice
from collections import defaultdict

import torch.multiprocessing as mp
import torch

# torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.utils import project_root, AverageMeter
from src.data.rulibrispeech import ruLibriSpeechDataset, TextEncoder, Collator, Featurizer
from src.data.utils import ru_alphabet
from src.metrics.metrics import WordErrorRate
import gc
# from matplotlib import pyplot as plt
import time
from src.model.squeezeformer.squeezeformer import Squeezeformer


def save_model(ckpt_path, model, optimizer, epoch, batch_index):
    ckpt = {
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    
def load_model(ckpt_path):
    return torch.load(ckpt_path, map_location=torch.device("cpu"))


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    ckpt_path = project_root()/"ckpt/squeezeformer"

    sample_rate=16000
    n_mels=64
    # data_path = pathlib.Path("/mnt/d") / "data/ruLibriSpeech/ruls_data"
    data_path = project_root() / "data/ruLibriSpeech/ruls_data"
    train_dataset = ruLibriSpeechDataset(data_path, "train", sample_rate=sample_rate)
    validation_dataset = ruLibriSpeechDataset(data_path, "test", sample_rate=sample_rate)


    encoder = TextEncoder(ru_alphabet)
    collator = Collator(encoder, max_length=50000)


    batch_size = 2

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collator,
        num_workers=0, pin_memory=False
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size,
        collate_fn=collator,
        num_workers=0, pin_memory=False
    )


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("train_dataset", len(train_dataset))
    print("validation_dataset", len(validation_dataset))
    featurizer = Featurizer(n_mels=n_mels, sample_rate=sample_rate).to(device)

    numclasses = len(ru_alphabet)

    model = Squeezeformer(
        num_classes=numclasses,
        input_dim=n_mels,
        encoder_dim=512
    ).to(device)
    criterion = nn.CTCLoss(zero_infinity=True, reduction='mean').to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.9995),  weight_decay=5e-6)

    storage = defaultdict(list)
    start_epoch = 0
    num_epoch = 10
    save_step = 500
    if os.path.exists(ckpt_path / "ckpt.pt"):
        try:
            ckpt = load_model(ckpt_path / "ckpt.pt")
            print("epoch", ckpt['epoch'], "batch_index", ckpt['batch_index'])
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("model and optimizer loaded")
        except Exception as e:
            print("couldn't load model", e)


    for epoch in range(start_epoch, num_epoch):
        train_loss_meter = AverageMeter()

        model.train()
        #for i, batch in enumerate(pbar := tqdm(islice(train_dataloader, 500))):
        for i, batch in enumerate(pbar := tqdm(train_dataloader)):
            optimizer.zero_grad()
            # Move batch to device if device != 'cpu'
            wav = batch['wav'].to(device)
            length = batch['length'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            mel, mel_length = featurizer(wav, length)
            mel = torch.swapaxes(mel, 1, 2).contiguous()
            
    #         print("max mel", mel.max())
    #         print("min mel", mel.min())
            
            outputs, output_lengths = model(mel, mel_length)
            
            loss = criterion(outputs.transpose(0, 1).contiguous(), targets, output_lengths, target_lengths)
            try:
                loss.backward()
                optimizer.step()

                l = loss.detach().cpu().numpy()
            except Exception as e:
                print("outputs", outputs.shape)
                print("output_lengths", output_lengths.shape)
                print("mel", mel.shape)
                print("mel_length", mel_length.shape)
                print("targets", targets.shape)
                print("target_lengths", target_lengths.shape)
                print(e)
    #         print("max grad", max(p.grad.max() for p in list(model.parameters())))
    #         print("min grad", max(p.grad.min() for p in list(model.parameters())))
            pbar.set_description(f"loss: {l:.5}")
            pbar.refresh()
            
            train_loss_meter.update(l)
            
            if i % save_step == 0 and i > 0:
                save_model(ckpt_path / "ckpt.pt", model, optimizer, epoch, i)
            
            # gc.collect()
            # torch.cuda.empty_cache()
            # time.sleep(0.1)
        storage['train_loss'].append(train_loss_meter.avg)

        validation_loss_meter = AverageMeter()
        validation_wer_meter = AverageMeter()

        model.eval()
        wer_metrics = WordErrorRate(encoder)
        for i, batch in enumerate(tqdm(islice(validation_dataloader, 1))):
            # gc.collect()
            # Move batch to device if device != 'cpu'
            wav = batch['wav'].to(device)
            length = batch['length'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            with torch.no_grad():
                mel, mel_length = featurizer(wav, length)
                mel = torch.swapaxes(mel, 1, 2).contiguous()
            
                outputs, output_lengths = model(mel, mel_length)
                loss = criterion(outputs.transpose(0, 1).contiguous(), targets, output_lengths, target_lengths)
                
            l = loss.detach().cpu().numpy()
            validation_loss_meter.update(l)
            validation_wer_meter.update(wer_metrics(targets.cpu(),  outputs.argmax(dim=-1).cpu()))
            
            pbar.set_description(f"loss: {l:.5}, wer: {validation_wer_meter.val}")
            pbar.refresh()
            # time.sleep(0.1)

        storage['validation_loss'].append(validation_loss_meter.avg)
        storage['validation_wer'].append(validation_wer_meter.avg)

        print(f"train_loss: {storage['train_loss']}, val_loss: {storage['validation_loss']}, val_wer: {storage['validation_wer']}")
        if epoch % 2 == 0:
            save_model(ckpt_path / f"ckpt_{epoch}.pt", model, optimizer, epoch, i)
        # fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # axes[0].plot(storage['train_loss'], label='train_loss')
        # axes[1].plot(storage['validation_loss'], label='validation_loss')
        # axes[2].plot(storage['validation_wer'], label='validation_WER')

        # for i in range(3):
        #     axes[i].grid()
        #     axes[i].legend()

        # plt.show()
    save_model(ckpt_path / f"ckpt_final.pt", model, optimizer, 0, 0)
