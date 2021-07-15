import torch
import torchvision as tv
import torch_xla
import torch_xla.core.xla_model as xm
import itertools as it

from typing import Tuple
from pathlib import Path
from pytpurch.models import WideResNet
from pytpurch.utils import Timer
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

CIFAR_10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}


def get_cifar10_dataloaders(
    batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    datadir = str(Path() / "__pycache__")
    transform_train = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(**CIFAR_10_STATS),
        ]
    )
    train = tv.datasets.CIFAR10(
        datadir, train=True, transform=transform_train, download=True
    )

    transform_test = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(**CIFAR_10_STATS),
        ]
    )
    val = tv.datasets.CIFAR10(
        datadir, train=False, transform=transform_test, download=True
    )
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers),
    )


def train(
    lr: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 200,
    num_workers: int = 2,
):
    device = xm.xla_device()
    model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.1
    )
    data_train, data_val = get_cifar10_dataloaders(batch_size, num_workers)

    print("Warming up model")
    for imgs, _ in it.islice(data_train, 5):
        model(imgs.to(device))

    print("Starting training")
    for epoch in range(num_epochs):

        model.train()
        train_desc = "Epoch [{epoch}/{num_epochs}] Iter[{n_iter}/{num_iters}]\t\tLoss: {loss:.4f}"

        with Timer() as train_timer:
            for n_iter, (imgs, labels) in enumerate(data_train):
                logits = model(imgs.to(device))
                loss = torch.nn.functional.cross_entropy(logits, labels.to(device))

                optimizer.zero_grad()
                loss.backward()
                xm.optimizer_step(optimizer, barrier=True)

                if n_iter % 10 == 0:
                    desc = train_desc.format(
                        epoch=epoch,
                        num_epochs=num_epochs,
                        n_iter=n_iter,
                        num_iters=len(data_train),
                        loss=loss.item(),
                    )
                    print(desc)

        scheduler.step()

        loss = 0.0
        correct = 0
        n_imgs = 0

        model.eval()
        val_desc = "Epoch [{epoch}/{num_epochs}] Iter[{n_iter}/{num_iters}]\t\tVal Loss: {loss:.4f} \t Val Accuracy: {acc:.2f}"
        with torch.no_grad(), Timer() as val_timer:
            for n_iter, (imgs, labels) in enumerate(data_val):
                logits = model(imgs.to(device))
                labels = labels.to(device)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                loss += torch.nn.functional.cross_entropy(logits, labels).item()
                n_imgs += len(imgs)

                if n_iter % 10 == 0:
                    desc = val_desc.format(
                        epoch=epoch,
                        num_epochs=num_epochs,
                        n_iter=n_iter,
                        num_iters=len(data_train),
                        loss=loss / n_imgs,
                        acc=correct / n_imgs,
                    )
                    print(desc)

        print("=========================================")
        print(
            f"Epoch [{epoch}/{num_epochs}  Val Loss: {loss / n_imgs}, Val Acc: {correct / n_imgs}"
        )
        print(f"  Training epoch: {train_timer.result}, val epoch: {val_timer.result}")


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    app.command()(train)
    app()
