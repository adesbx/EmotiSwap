import optuna
import torch
import time
from torch import Tensor
from torch.nn import Module, Linear, Conv2d, AdaptiveAvgPool2d, Flatten, BatchNorm2d, Dropout
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torchvision.models import resnet18
from optuna.trial import TrialState
from optuna import Trial
from core import core
from collections import Counter

import torch.nn.functional as F

# writer = SummaryWriter()

class CnnEmotion(Module):
    """Convolutional neural network

    Args:
        Module (nn.Module): basic class of neural network
    """
   
    def __init__(self, dropout=0.4):
        """Constructor
        
        """
        super(CnnEmotion, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = Conv2d(128, 256, kernel_size=(3, 3), padding=1)

        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn4 = BatchNorm2d(256)

        self.adavgpool = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(dropout)

        self.fc1 = Linear(256, 256)
        self.fc2 = Linear(256, 128)
        self.output = Linear(128, 7)

    def forward(self, x: Tensor) -> Tensor:
        """Process of a neuron in the model

        Args:
            x (Tensor): the batch of data selected

        Returns:
            Tensor: output of the neuron
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.adavgpool(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.output(x)

        return output
    
def compute_class_weight(dataset: TensorDataset, num_classes: int) -> Tensor: 
    targets = [label for _, label in dataset]

    counts  = Counter(targets)

    total = len(targets)
    weight = []

    for i in range(num_classes):
        weight.append(total / (num_classes * counts[i]))

    return torch.tensor(weight, dtype=torch.float)

    
def objective(trial: Trial) -> float:
    """Function to maximise

    Args:
        trial (Trial): For suggest hyperparameter

    Returns:
        float: accuracy of the model tuned
    """
    global test_dataset_g, model_to_use
    if model_to_use == "cnn":
        model = CnnEmotion()
    elif model_to_use == "resnet":
        model = resnet18(weights="IMAGENET1K_V1")
        model.conv1 = Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = Linear(model.fc.in_features, 7)
    print("MODEL USED: ", model_to_use)
    model.to(device)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_dataset, val_dataset, test_dataset = core.load_split_data()
    test_dataset_g = test_dataset
    batch_size = trial.suggest_int("batch", 8, 32) 
    weight = compute_class_weight(train_dataset, 7)
    weight.to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight, label_smoothing=0.1)
    loss_func.to(device)
    # Training of the model.
    start_time = time.time()
    model_trained, nb_epoch, local_loss_mean, accuracy = core.training_early_stopping(
        model, train_dataset, val_dataset, batch_size, loss_func, optimizer, trial
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    model_info = [
        model_trained,
        accuracy,
        local_loss_mean,
        elapsed_time,
        batch_size,
        lr,
        nb_epoch,
    ]
    trial.set_user_attr(key="model", value=model_trained)
    # with open("./csv/dataMlp2.csv", "a", newline="") as csvfile:
    #     spamwriter = csv.writer(csvfile)
    #     if csvfile.tell() == 0:
    #         spamwriter.writerow(
    #             [
    #                 "Accuracy",
    #                 "Validation Loss",
    #                 "Elapsed time",
    #                 "Batch Size",
    #                 "Learning Rate",
    #                 "Epochs",
    #             ]
    #         )
    #     spamwriter.writerow(model_info[1:])
    # writer.add_hparams(
    #     {
    #         "lr": lr,
    #         "batch_size": batch_size,
    #         "nb_epoch": nb_epoch,
    #     },
    #     {
    #         "hparam/Accuracy": accuracy,
    #         "hparam/Validation Loss": local_loss_mean,
    #         "hparam/time": elapsed_time,
    #     },
    # )
    torch.cuda.empty_cache()
    return accuracy

if __name__ == "__main__":
    global model_to_use
    model_to_use = "resnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu" # force pour passer en cpu only
    print(f"Using device: {device}")


    core = core(device=device)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20,  timeout=5000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    core.final_test(trial.user_attrs["model"], test_dataset_g)

    if model_to_use == "cnn":
        torch.save(trial.user_attrs["model"].state_dict(), "./assets/models/model_cnn.pth")
    elif model_to_use == "resnet":
        torch.save(trial.user_attrs["model"].state_dict(), "./assets/models/model_resnet.pth")
