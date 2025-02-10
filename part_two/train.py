from pathlib import Path
from typing import Callable
import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import logging

logging.basicConfig(level=logging.INFO)


class TrainerBase:
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        device: str,
        checkpoint_path: Path = None,
        resume: bool = False,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.epoch = 0

        if resume and checkpoint_path:
            self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_path and self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint.get("epoch", 0)
            print(
                f"Checkpoint loaded from {self.checkpoint_path}, starting at epoch {self.epoch}."
            )
        else:
            print("No checkpoint found. Training from scratch.")

    def _save_checkpoint(self, epoch):
        if self.checkpoint_path:
            state = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            checkpoint_file = self.checkpoint_path / f"checkpoint_{epoch + 1}.pth"
            torch.save(state, checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")

    def train(self, train_loader: DataLoader, num_epochs: int):
        self.model.train()
        for epoch in range(self.epoch, num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = (
                    data["image"].to(self.device),
                    data["label"].to(self.device),
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 2000 == 0:
                    print(
                        f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 2000:.4f}"
                    )
                    running_loss = 0.0

            self._save_checkpoint(epoch)

    def test(self, test_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = (
                    data["image"].to(self.device),
                    data["label"].to(self.device),
                )
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, path: Path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")


class TrainerClassifier(TrainerBase):
    pass


class TrainerNER(TrainerBase):
    def train(self, train_loader: DataLoader, num_epochs: int):
        self.model.train()
        for epoch in range(self.epoch, num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels, text = (
                    data["image"].to(self.device),
                    data["label"].to(self.device),
                    data["text"].to(self.device),
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs, text)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 2000 == 0:
                    logging.info(
                        f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 2000:.4f}"
                    )
                    running_loss = 0.0

            self._save_checkpoint(epoch)

    def test(self, test_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels, text = (
                    data["image"].to(self.device),
                    data["label"].to(self.device),
                    data["text"].to(self.device),
                )
                outputs = self.model(images, text)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
