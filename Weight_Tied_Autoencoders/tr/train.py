import torch
from torch import nn
import tqdm


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim,
                 device: str = "cpu:0"
                 ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.model
        self.optimizer = optimizer

    def train_step(self,
                   dataloader: torch.utils.data.DataLoader
                   ):
        self.model.train()
        step_loss = 0
        n = 0
        for X, mask in dataloader:
            X = X.to(self.device)
            mask = mask.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, mask)
            step_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            n += 1
        return step_loss / n

    def test_step(self,
                  dataloader: torch.utils.data.DataLoader,
                  ):
        self.model.eval()
        step_loss = 0
        n = 0
        for X, mask in dataloader:
            X = X.to(self.device)
            mask = mask.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, mask)
            step_loss += loss.item()
            n += 1
        return step_loss / n

    def fit(self,
            train_dataloder: torch.utils.data.DataLoader,
            test_dataloder: torch.utils.data.DataLoader,
            output_freq: int = 2,
            first_step: int = 0,
            epochs: int = 10):
        last_step = first_step + epochs
        avg_train_loss = {}
        avg_val_loss = {}
        for epoch in tqdm.tqdm(range(epochs)):
            train_loss = self.train_step(train_dataloder)
            test_loss = self.test_step(test_dataloder)
            avg_train_loss[first_step + epoch] = train_loss
            avg_val_loss[first_step + epoch] = test_loss
            if epoch % output_freq == 0:
                print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                    Train loss: {train_loss:.4f} Validation loss: {test_loss:.4f}")
        return avg_train_loss, avg_val_loss
