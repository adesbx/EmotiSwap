import torch
import optuna
import json
from optuna import Trial
from torch import Tensor
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, MSELoss, CrossEntropyLoss
from torchvision import datasets, transforms

# writer = SummaryWriter()


class core:
    """Implement generalisation methods, for all models"""

    def __init__(self, device):
        """Constructor"""
        self.device = device

    def load_split_data(self, percentage: list[int] = [0.8, 0.2]) -> tuple:
        """Load from the path and split dataset in train, validation and test part

        Args:
            percentage (list[int], optional): percentage of repartition for train and validation. Defaults to [0.8, 0.2].

        Returns:
            tuple: train, val, test dataset
        """
        transform = transforms.Compose([
            transforms.Resize((96,96)), # if using cnn may need to back this to 48,48
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(96, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        train_dataset = datasets.ImageFolder(root="./assets/img/archive/train", transform=transform)
        train_dataset, val_dataset = random_split(train_dataset, percentage)
        test_dataset = datasets.ImageFolder(root="./assets/img/archive/test", transform=transform)


        class_to_idx = train_dataset.dataset.class_to_idx 

        with open("./assets/models/class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f, indent=4)

        return train_dataset, val_dataset, test_dataset

    def train_step(
        self,
        model: Module,
        train_loader: DataLoader,
        loss_func: MSELoss | CrossEntropyLoss,
        optim: SGD | Adam,
        epoch: int,
    ):
        """Train the model from train_loader, adjusted weight

        Args:
            model (Module): pytorch model
            train_loader (DataLoader): train loader
            loss_func (MSELoss | CrossEntropyLoss): goal is to min this function
            optim (SGD | Adam): optimise the gradient descent
            epoch (int): epoch of this train step for write the loss/train
        """
        model.train()
        running_loss = 0.0
        i = 0
        for x, t in train_loader:
            x, t = x.to(self.device), t.to(self.device)
            y = model(x)

            loss = loss_func(y, t) # try to change the parameters from t, y -> y, t
            loss.backward()
            optim.step()
            optim.zero_grad()
            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...
                # ...log the running loss
                # writer.add_scalar(
                #     "loss/train", running_loss / 1000, epoch * len(train_loader) + i
                # )
                running_loss = 0.0
            i += 1

    def validation_step(
        self,
        model: Module,
        val_loader: DataLoader,
        loss_func: MSELoss | Adam,
        epoch: int,
    ):
        """Validate the model from val_loader, doesn't move weight

        Args:
            model (Module): pytorch model
            val_loader (DataLoader): validation loader
            loss_func (MSELoss | CrossEntropyLoss): goal is to min this function
            optim (SGD | Adam): optimise the gradient descent
            epoch (int): epoch of this train step for write the loss/train
        """
        loss_total = 0
        acc = 0.0
        total_samples = 0
        i = 0.0
        running_loss = 0.0
        model.eval()  
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(self.device), t.to(self.device)
                y = model(x)
                loss = loss_func(y, t)
                loss_total += loss
                # acc += (torch.argmax(y, 1) == torch.argmax(t, 1)).sum().item()
                acc += (torch.argmax(y, dim=1) == t).sum().item()
                total_samples += t.size(0)
                running_loss += loss.item()
                if i % 1000 == 999:  # every 1000 mini-batches...
                    # ...log the running loss
                    # writer.add_scalar(
                    #     "loss/validation", running_loss / 1000, epoch * len(val_loader) + i
                    # )
                    running_loss = 0.0
                i += 1
            return loss_total / len(val_loader), acc / total_samples

    def training_early_stopping(
        self,
        model: Module,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        batch_size: int,
        loss_func: MSELoss | CrossEntropyLoss,
        optim: MSELoss | Adam,
        trial: Trial = None,
        max_epochs: int = 20,
        min_delta: int = 0.0005,
        patience: int = 10,
    ) -> tuple:
        """Training phase, with early-stopping if no more convergence

        Args:
            model (Module): pytorch model
            train_dataset (TensorDataset):
            val_dataset (TensorDataset):
            batch_size (int): size of batch that is pick in the dataloader
            loss_func (MSELoss | CrossEntropyLoss): function to minimize
            optim (MSELoss | Adam): optimise the gradient descent
            trial (Trial, optional): Object from optuna for manage the hyper parameter choice. Defaults to None.
            max_epochs (int, optional): limit of a train_step call. Defaults to 100.
            min_delta (int, optional): minimum of improvement before lossing patience . Defaults to 0.0005.
            patience (int, optional): decrease until 0 for early-stopping. Defaults to 10.

        Raises:
            optuna.exceptions.TrialPruned: Pruned by algo from optuna

        Returns:
            tuple: model, number of epoch, loss, accuracy
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        previous_loss_mean = 1
        previous_improvement = 1
        for n in range(max_epochs):
            self.train_step(model, train_loader, loss_func, optim, n)
            local_loss_mean, accuracy = self.validation_step(
                model, val_loader, loss_func, n
            )
            improvement = previous_loss_mean - local_loss_mean
            if previous_improvement == improvement or improvement <= min_delta:
                if patience == 0:
                    previous_loss_mean = local_loss_mean
                    print(
                        f"early-stopped at {n} epochs for {previous_loss_mean} loss_mean"
                    )
                    break
                patience -= 1
            previous_loss_mean = local_loss_mean
            previous_improvement = improvement
            if trial:
                trial.report(accuracy, n)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            # writer.flush()
        return model, n, previous_loss_mean.item(), accuracy

    def final_test(self, best_model: Module, test_dataset: TensorDataset):
        """Final test for the best of the best model

        Args:
            best_model (Module): the best pytorch model trained
            test_dataset (TensorDataset):
        """
        acc = 0.0
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for x, t in test_loader:
            x, t = x.to(self.device), t.to(self.device)
            y = best_model(x)
            acc += (torch.argmax(y, dim=1) == t).sum().item()
        print(acc / len(test_dataset))



