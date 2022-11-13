import torch
from torchvision.transforms import transforms

# datasets
import data.data_utils as data_utils

# models
from models.PerFedAvg_Digit5 import SimpleDigit5CNNModel


def setup_datasets(hp):
    dataset = hp.dataset
    # load the data and split it among clients
    if dataset == 'digit5':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_digit5(hp=hp, verbose=True)
    elif dataset == 'office':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office(hp=hp, verbose=True)
    elif dataset == 'home':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office_home(hp=hp, verbose=True)
    else:
        assert dataset == 'pacs'
        train_loaders, test_loaders, public_loader, stats = data_utils.get_pacs(hp=hp, verbose=True)
    users = [i for i in range(len(train_loaders))]
    return users, train_loaders, test_loaders


def select_model(algorithm, model_name):
    if algorithm == 'perfedavg':
        if model_name == 'SimpleDigit5CNNModel':
            model = SimpleDigit5CNNModel()
        elif model_name == 'home':
            pass
        else:
            print(f"Unimplemented model {model_name} for algorithm {algorithm}.")
    return model


def fed_average(updates):
    total_weight = 0
    (client_samples_num, new_params) = updates[0][0], updates[0][1]

    for item in updates:
        (client_samples_num, client_params) = item[0], item[1]
        total_weight += client_samples_num

    for k in new_params.keys():
        for i in range(0, len(updates)):
            client_samples, client_params = updates[i][0], updates[i][1]
            # weight
            w = client_samples / total_weight
            if i == 0:
                new_params[k] = client_params[k] * w
            else:
                new_params[k] += client_params[k] * w
    # return global model params
    return new_params


def avg_metric(metricList):
    total_weight = 0
    total_metric = 0
    for (samples_num, metric) in metricList:
        total_weight += samples_num
        total_metric += samples_num * metric
    average = total_metric / total_weight

    return average
