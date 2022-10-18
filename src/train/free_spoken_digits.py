import sys
sys.path.append("..")

import pathlib
from tqdm import tqdm
from itertools import islice
from collections import defaultdict

import torch.multiprocessing as mp
mp.set_start_method('spawn')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.utils import project_root
from data.free_spoken_digits import FreeSpokenDigits, Collator, Featurizer
from src.model.free_spoken_digits import Model


dataset = FreeSpokenDigits(
    project_root()/"data/free_spoken_digits/free-spoken-digit-dataset-master"
)

train_ratio = 0.9
train_size = int(len(dataset) * train_ratio)
validation_size = len(dataset) - train_size

indexes = torch.randperm(len(dataset))
train_indexes = indexes[:train_size]
validation_indexes = indexes[train_size:]

train_dataset = Subset(dataset, train_indexes)
validation_dataset = Subset(dataset, validation_indexes)


train_dataloader = DataLoader(
    train_dataset, batch_size=32,
    shuffle=True, collate_fn=Collator(),
    num_workers=2, pin_memory=True
)

validation_dataloader = DataLoader(
    validation_dataset, batch_size=32,
    collate_fn=Collator(),
    num_workers=2, pin_memory=True
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Model(input_dim=64, hidden_size=128).to(device)
featurizer = Featurizer().to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    storage = defaultdict(list)
    num_epoch = 10

    for epoch in range(num_epoch):
        train_loss_meter = AverageMeter()

        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            # Move batch to device if device != 'cpu'
            wav = batch['wav'].to(device, non_blocking=True)
            length = batch['length'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            mel, mel_length = featurizer(wav, length)
            output = model(mel, mel_length)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        storage['train_loss'].append(train_loss_meter.avg)

        validation_loss_meter = AverageMeter()
        validation_accuracy_meter = AverageMeter()

        model.eval()
        for i, batch in enumerate(tqdm(validation_dataloader)):
            # Move batch to device if device != 'cpu'
            wav = batch['wav'].to(device)
            length = batch['length'].to(device)
            label = batch['label'].to(device)

            with torch.no_grad():

                mel, mel_length = featurizer(wav, length)
                output = model(mel, mel_length)

                loss = criterion(output, label)

            matches = (output.argmax(dim=-1) == label).float().mean()

            validation_loss_meter.update(loss.item())
            validation_accuracy_meter.update(matches.item())

        storage['validation_loss'].append(validation_loss_meter.avg)
        storage['validation_accuracy'].append(validation_accuracy_meter.avg)

        print(f"train_loss: {storage['train_loss']}, val_loss: {storage['validation_loss']}, val_accuracy: {storage['validation_accuracy']}")

    model_path = project_root()/"ckpt/free_spoken_digits.ckpt"
    torch.save(model.eval(), model_path)