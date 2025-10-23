import os
import time
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type  =int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class Logger:
    def __init__(self, filepath):
        self.logs = []
        self.filepath = filepath

    def log(self, message):
        self.logs.append(message)
    def write(self):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            for log in self.logs:
                f.write(log + '\n')

class Indexed(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return idx, x, y


@torch.no_grad()
def evaluate(model, eval_loader, gpu):
    """Distributed evaluation that returns global accuracy across all ranks."""
    model.eval()

    correct = torch.tensor(0, dtype=torch.long, device=gpu)
    total   = torch.tensor(0, dtype=torch.long, device=gpu)

    for batch in eval_loader:
        # your train dataset yields (idx, x, y). We'll handle both shapes.
        if len(batch) == 3:
            _, images, labels = batch
        else:
            images, labels = batch

        images = images.to(gpu, non_blocking=True)
        labels = labels.to(gpu, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum()
        total   += labels.numel()

    # aggregate across processes
    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total,   op=dist.ReduceOp.SUM)

    acc = (correct.float() / total.float()).item()
    model.train()  # restore for training
    return acc

def train(gpu, args):

    # logger = Logger(Path(f'./logs/log_single_gpu_{gpu}.txt'))
    main_logger = Logger(Path(f'./logs/single_gpu/main_log.txt'))

    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 128
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    train_dataset = Indexed(train_dataset)
    # train_dataset = Subset(train_dataset, range(20))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    start = time.perf_counter()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (idx, images, labels) in enumerate(train_loader):
            
            # logger.log(f"{" ".join(map(str, idx.tolist()))}")
            # import ipdb; ipdb.set_trace()

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))

        acc = evaluate(model, train_loader, gpu) 
        # logger.log("")
        main_logger.log(f"epoch [{epoch+1}/{args.epochs}] | accuracy: {acc:.4f}")

    msg = "\ntraining complete in: " + str(time.perf_counter() - start)
    print(msg)
    main_logger.log(msg)
    main_logger.write()

    # logger.write()
    

if __name__ == '__main__':
    main()