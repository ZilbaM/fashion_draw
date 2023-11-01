from __future__ import annotations

from itertools import cycle
from torch.optim import Optimizer
from torch.utils.data import (DataLoader, Dataset)
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

class FashionMNISTDataset(Dataset):
    def __init__(self, train: bool, path: str, device: torch.device) -> None:
        super().__init__()
        self.path = path
        self.prefix = 'train' if train else 'test'
        self.path_xs = os.path.join(self.path, f'FashionMNIST_{self.prefix}_xs.pt')
        self.path_ys = os.path.join(self.path, f'FashionMNIST_{self.prefix}_ys.pt')
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])

        if not os.path.exists(self.path_xs) or not os.path.exists(self.path_ys):
            set = FashionMNIST(path, train=train, download=True, transform=self.transform)
            loader = DataLoader(set, batch_size=batch_size, shuffle=train)
            n = len(set)

            xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
            ys = torch.empty((n, ), dtype=torch.long)
            desc = f'Preparing {self.prefix.capitalize()} Set'
            for i, (x, y) in enumerate(tqdm(loader, desc=desc)):
                xs[i * batch_size:min((i + 1) * batch_size, n)] = x
                ys[i * batch_size:min((i + 1) * batch_size, n)] = y

            torch.save(xs, self.path_xs)
            torch.save(ys, self.path_ys)
        
        self.device = device
        self.xs = torch.load(self.path_xs, map_location=self.device)
        self.ys = torch.load(self.path_ys, map_location=self.device)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class FeaturesDector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = FeaturesDector(1, 64)
        self.fc = MLP(3136, 10)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler, epochs: int) -> None:
        self.train()
        batches = iter(cycle(loader))
        for _ in tqdm(range(epochs * len(loader)), desc='Training'):
            x, l = next(batches)
            optimizer.zero_grad(set_to_none=True)
            logits = self(x)
            loss = F.nll_loss(torch.log_softmax(logits, dim=1), l)
            loss.backward()
            optimizer.step()
            scheduler.step()

    @torch.inference_mode()
    def test(self, loader: DataLoader) -> None:
        self.eval()
        loss, acc = 0, 0.0
        for x, l in tqdm(loader, total=len(loader), desc='Testing'):
            logits = self(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            loss += F.nll_loss(torch.log_softmax(logits, dim=1), l, reduction='sum').item()
            acc += (preds == l.view_as(preds)).sum().item()
        testLoss = (loss / len(loader.dataset))
        print()
        print(f'Loss: {testLoss}')
        print(f'Accuracy: {acc / len(loader.dataset) * 100:.2f}%')
        print()
        return testLoss


if __name__ == '__main__':
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    import torch.onnx as tonnx


    device = torch.device('cpu')
    epochs = 3
    batch_size = 256
    lr = 1e-2

    train_set = FashionMNISTDataset(True, '/tmp/datasets', device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_set = FashionMNISTDataset(False, '/tmp/datasets', device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    bestLoss = 9999
    keptModel : CNN
    for i in range(5):
        model = CNN().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.7, 0.9)) 
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=int(((len(train_set) - 1) // batch_size + 1) * epochs))
        model.fit(train_loader, optimizer, scheduler, epochs)
        testLoss = model.test(test_loader)
        if i == 0:
            keptModel = model
        elif bestLoss>testLoss :
            bestLoss = testLoss
            keptModel = model
            torch.save(model.state_dict(), 'FashionMNIST_cnn.pt')
            print(f'best loss found : {testLoss:.2e}')

    
    tonnx.export(
        keptModel.cpu(),
        torch.empty((1, 1, 28, 28), dtype=torch.float32),
        'fashionMNIST.onnx',
        export_params=True,
        opset_version=9,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )