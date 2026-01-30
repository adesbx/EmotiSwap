import optuna
import torch
import time
from torch import Tensor
from torch.nn import Module, Linear, Conv2d
from torch.utils.tensorboard import SummaryWriter
from optuna.trial import TrialState
from optuna import Trial
from core import core

import torch.nn.functional as F

# writer = SummaryWriter()

class CnnEmotion(Module):
    """Convolutional neural network

    Args:
        Module (nn.Module): basic class of neural network
    """
   
    def __init__(self):
        """Constructor
        
        """
        super(CnnEmotion, self).__init__()
        self.conv1 = Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = Conv2d(6, 16, kernel_size=(5, 5))

        self.fc1 = Linear(16 * 9 * 9, 120)
        self.fc2 = Linear(120, 84)

        self.output = Linear(84, 7)

    def forward(self, x: Tensor) -> Tensor:
        """Process of a neuron in the model

        Args:
            x (Tensor): the batch of data selected

        Returns:
            Tensor: output of the neuron
        """
        x = x.view(-1, 1, 48, 48)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.output(x)

        return output
    
def objective(trial: Trial) -> float:
    """Function to maximise

    Args:
        trial (Trial): For suggest hyperparameter

    Returns:
        float: accuracy of the model tuned
    """
    global test_dataset_g
    model = CnnEmotion()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_dataset, val_dataset, test_dataset = core.load_split_data()
    test_dataset_g = test_dataset
    batch_size = trial.suggest_int("batch", 3, 100)
    loss_func = torch.nn.CrossEntropyLoss()
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
    return accuracy

if __name__ == "__main__":
    core = core()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000,  timeout=10800)

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